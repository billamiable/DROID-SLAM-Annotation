import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from modules.extractor import BasicEncoder
from modules.corr import CorrBlock
from modules.gru import ConvGRU
from modules.clipping import GradientClip

from lietorch import SE3
from geom.ba import BA

import geom.projective_ops as pops
from geom.graph_utils import graph_to_edge_list, keyframe_indicies

from torch_scatter import scatter_mean


def cvx_upsample(data, mask):
    """ upsample pixel-wise transformation field """
    batch, ht, wd, dim = data.shape
    data = data.permute(0, 3, 1, 2)
    mask = mask.view(batch, 1, 9, 8, 8, ht, wd)
    mask = torch.softmax(mask, dim=2)

    up_data = F.unfold(data, [3,3], padding=1)
    up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)

    up_data = torch.sum(mask * up_data, dim=2)
    up_data = up_data.permute(0, 4, 2, 5, 3, 1)
    up_data = up_data.reshape(batch, 8*ht, 8*wd, dim)

    return up_data

def upsample_disp(disp, mask):
    batch, num, ht, wd = disp.shape
    disp = disp.view(batch*num, ht, wd, 1)
    mask = mask.view(batch*num, -1, ht, wd)
    return cvx_upsample(disp, mask).view(batch, num, 8*ht, 8*wd)

# TODO GraphAgg?
class GraphAgg(nn.Module):
    def __init__(self):
        super(GraphAgg, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.eta = nn.Sequential(
            nn.Conv2d(128, 1, 3, padding=1),
            GradientClip(),
            nn.Softplus())

        self.upmask = nn.Sequential(
            nn.Conv2d(128, 8*8*9, 1, padding=0))

    def forward(self, net, ii):
        batch, num, ch, ht, wd = net.shape
        net = net.view(batch*num, ch, ht, wd)

        _, ix = torch.unique(ii, return_inverse=True)
        net = self.relu(self.conv1(net))

        net = net.view(batch, num, 128, ht, wd)
        net = scatter_mean(net, ix, dim=1)
        net = net.view(-1, 128, ht, wd)

        net = self.relu(self.conv2(net))

        eta = self.eta(net).view(batch, -1, ht, wd)
        upmask = self.upmask(net).view(batch, -1, 8*8*9, ht, wd)

        return .01 * eta, upmask


class UpdateModule(nn.Module):
    def __init__(self):
        super(UpdateModule, self).__init__()
        cor_planes = 4 * (2*3 + 1)**2

        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, 128, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True))

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(4, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True))

        self.weight = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip(),
            nn.Sigmoid())

        self.delta = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip())

        self.gru = ConvGRU(128, 128+128+64)
        self.agg = GraphAgg()

    '''
        Update Operator
            Core design in RAFT, mimics the steps of an optimization algorithm
    '''
    def forward(self, net, inp, corr, flow=None, ii=None, jj=None):
        """ RaftSLAM update operator """

        batch, num, ch, ht, wd = net.shape

        if flow is None:
            flow = torch.zeros(batch, num, 4, ht, wd, device=net.device)

        output_dim = (batch, num, -1, ht, wd)
        net = net.view(batch*num, -1, ht, wd)
        inp = inp.view(batch*num, -1, ht, wd)        
        corr = corr.view(batch*num, -1, ht, wd)
        flow = flow.view(batch*num, -1, ht, wd)

        # Step1: use nn to process correlation and flow map before feeding into gru
        corr = self.corr_encoder(corr)
        flow = self.flow_encoder(flow)

        '''
            ConvGRU Module
                Gated activation unit based on the GRU cell
        '''
        # Step2: use gru module to perform iterative update and obtain updated hidden state
        net = self.gru(net, inp, corr, flow)

        # Step3: pass updated hidden state to conv layers to predict the flow update (delta_flow)
        ### update variables ###
        delta = self.delta(net).view(*output_dim)
        weight = self.weight(net).view(*output_dim)

        delta = delta.permute(0,1,3,4,2)[...,:2].contiguous()
        weight = weight.permute(0,1,3,4,2)[...,:2].contiguous()

        # Step4: TODO unpack output features from gru module
        net = net.view(*output_dim)

        if ii is not None:
            # Step5: TODO obtain 8*8 mask for inverse depth upsampling. what's eta and self.agg?
            # eta - pixelwise damping factor?
            eta, upmask = self.agg(net, ii.to(net.device))
            return net, delta, weight, eta, upmask

        else:
            return net, delta, weight


class DroidNet(nn.Module):
    def __init__(self):
        super(DroidNet, self).__init__()
        '''
            Feature encoder
                Sub-nn-module in RAFT
        '''
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')
        '''
            Context encoder
                Sub-nn-module in RAFT
        '''
        self.cnet = BasicEncoder(output_dim=256, norm_fn='none')
        self.update = UpdateModule()


    '''
        Feature extraction
            Simple NN layer
    '''
    def extract_features(self, images):
        """ run feature extraction networks """

        # normalize images
        images = images[:, :, [2,1,0]] / 255.0
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
        images = images.sub_(mean[:, None, None]).div_(std[:, None, None])

        fmaps = self.fnet(images)
        net = self.cnet(images)
        
        net, inp = net.split([128,128], dim=2)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        return fmaps, net, inp


    '''
        Frame-to-Frame Tracking - Major part for nn module of DRIOD-SLAM
            forward path for nn - define the computation performed at every call
    '''
    def forward(self, Gs, images, disps, intrinsics, graph=None, num_steps=12, fixedp=2):
        """ Estimates SE3 or Sim3 between pair of frames """

        # Step1: obtain correlated pairs of frames from graph based on optical flow distance
        u = keyframe_indicies(graph)
        ii, jj, kk = graph_to_edge_list(graph)

        ii = ii.to(device=images.device, dtype=torch.long)
        jj = jj.to(device=images.device, dtype=torch.long)

        # Step2: extract features from images
        # feature map from feature encoder (fmaps) and context encoder (net), latent hidden state (inp)
        fmaps, net, inp = self.extract_features(images)
        net, inp = net[:,ii], inp[:,ii] # TODO why this work? print shape

        # Step3: construct 4D correlation volumes to obtain correlation pyramid
        # TODO print shape, seems only correlated ones are involved for constructing 4D cost volume?
        corr_fn = CorrBlock(fmaps[:,ii], fmaps[:,jj], num_levels=4, radius=3)

        # Step4: use projection-based matching and lookup operator to obtain initial correspondences
        ht, wd = images.shape[-2:]
        # coords0, coords1 - 2D pixel coordinate (x,y)
        coords0 = pops.coords_grid(ht//8, wd//8, device=images.device)
        '''
            Loopup Operator
                Obtains 2D grid region that contains the potential correspondences
        '''
        coords1, _ = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
        target = coords1.clone()

        # Step5: feed prepared inputs to major part of network
        Gs_list, disp_list, residual_list = [], [], []
        for step in range(num_steps):
            Gs = Gs.detach()
            disps = disps.detach()
            coords1 = coords1.detach()
            target = target.detach()

            # Step5.1: extract motion-based flow features TODO what's motion?
            '''
                Loopup Operator
                    Operate on multi-level correlation pyramid
            '''
            corr = corr_fn(coords1)
            resd = target - coords1
            flow = coords1 - coords0

            motion = torch.cat([flow, resd], dim=-1) # concat
            motion = motion.permute(0,1,4,2,3).clamp(-64.0, 64.0) # rearrange ordering & limit value within range

            # Step5.2: feed into GRU to get correction value for correspondence and confidence value
            # Inputs: context feature (net), hidden state (inp), correlation feature (corr), flow feature (motion)
            # Outputs: correction term for correspondence field (delta) and associated confidence map (weight)
            net, delta, weight, eta, upmask = \
                self.update(net, inp, corr, motion, ii, jj)

            # Step5.3: get corrected corrspondence with initial correspondence and correction term
            target = coords1 + delta

            for i in range(2):
                '''
                    Deep Bundle Adjustment Layer
                        Core design in DRIOD-SLAM
                '''
                # Step5.4: feed into DBA layer to obtain delta value for pose and inverse depth
                # Inputs: corrected correspondence field (target) and associated confidence map (weight), 
                #         pixelwise damping factor (eta), initial pose (Gs) and inverse depth (disps)
                # Outputs: optimized pose (Gs) and inverse depth (disps)
                Gs, disps = BA(target, weight, eta, Gs, disps, intrinsics, ii, jj, fixedp=2)

            # Step5.5: obtain updated variable values and residual loss for next optimization iteration
            coords1, valid_mask = pops.projective_transform(Gs, disps, intrinsics, ii, jj) # same formulation
            residual = (target - coords1) # keep track of residual loss during iteration

            Gs_list.append(Gs)
            # Step5.6: use unfold func to upsample estimated optical flow to match the input image size
            disp_list.append(upsample_disp(disps, upmask))
            residual_list.append(valid_mask * residual)

        # Step5.7: output computed pose, inverse depth and residual error
        return Gs_list, disp_list, residual_list
