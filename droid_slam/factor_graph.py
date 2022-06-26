import torch
import lietorch
import numpy as np

import matplotlib.pyplot as plt
from lietorch import SE3
from modules.corr import CorrBlock, AltCorrBlock
import geom.projective_ops as pops


class FactorGraph:
    def __init__(self, video, update_op, device="cuda:0", corr_impl="volume", max_factors=-1, upsample=False):
        self.video = video
        self.update_op = update_op
        self.device = device
        self.max_factors = max_factors
        self.corr_impl = corr_impl
        self.upsample = upsample

        # operator at 1/8 resolution
        self.ht = ht = video.ht // 8
        self.wd = wd = video.wd // 8

        self.coords0 = pops.coords_grid(ht, wd, device=device)
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.age = torch.as_tensor([], dtype=torch.long, device=device)

        self.corr, self.net, self.inp = None, None, None
        self.damping = 1e-6 * torch.ones_like(self.video.disps)

        self.target = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

        # inactive factors
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.ii_bad = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_bad = torch.as_tensor([], dtype=torch.long, device=device)

        self.target_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

    def __filter_repeated_edges(self, ii, jj):
        """ remove duplicate edges """

        keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        eset = set(
            [(i.item(), j.item()) for i, j in zip(self.ii, self.jj)] +
            [(i.item(), j.item()) for i, j in zip(self.ii_inac, self.jj_inac)])

        for k, (i, j) in enumerate(zip(ii, jj)):
            keep[k] = (i.item(), j.item()) not in eset

        return ii[keep], jj[keep]

    def print_edges(self):
        ii = self.ii.cpu().numpy()
        jj = self.jj.cpu().numpy()

        ix = np.argsort(ii)
        ii = ii[ix]
        jj = jj[ix]

        w = torch.mean(self.weight, dim=[0,2,3,4]).cpu().numpy()
        w = w[ix]
        for e in zip(ii, jj, w):
            print(e)
        print()

    def filter_edges(self):
        """ remove bad edges """
        conf = torch.mean(self.weight, dim=[0,2,3,4])
        mask = (torch.abs(self.ii-self.jj) > 2) & (conf < 0.001)

        self.ii_bad = torch.cat([self.ii_bad, self.ii[mask]])
        self.jj_bad = torch.cat([self.jj_bad, self.jj[mask]])
        self.rm_factors(mask, store=False)

    def clear_edges(self):
        self.rm_factors(self.ii >= 0)
        self.net = None
        self.inp = None

    '''
        Add Edge to Factor Graph
            One of key part for factor graph class
    '''
    @torch.cuda.amp.autocast(enabled=True)
    def add_factors(self, ii, jj, remove=False):
        """ add edges to factor graph """

        # Step1: preprocessing of input into tensor
        # returns True if the specified object is of the specified type
        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        # Step2: remove duplicate edges
        ii, jj = self.__filter_repeated_edges(ii, jj)


        if ii.shape[0] == 0:
            return

        # Step3: place limit on number of factors
        if self.max_factors > 0 and self.ii.shape[0] + ii.shape[0] > self.max_factors \
                and self.corr is not None and remove:
            
            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            self.rm_factors(ix >= self.max_factors - ii.shape[0], store=True)

        # Step4: construct context feature for new edges TODO nets[ii]
        net = self.video.nets[ii].to(self.device).unsqueeze(0) # opposite of squeeze

        # Step5: constrcut 4D correlation volume for new edges
        if self.corr_impl == "volume":
            c = (ii == jj).long()
            fmap1 = self.video.fmaps[ii,0].to(self.device).unsqueeze(0) # feature map
            fmap2 = self.video.fmaps[jj,c].to(self.device).unsqueeze(0)
            corr = CorrBlock(fmap1, fmap2) # 4D correlation volum
            self.corr = corr if self.corr is None else self.corr.cat(corr)

            # Step6: constrcut hidden state for new edges
            inp = self.video.inps[ii].to(self.device).unsqueeze(0)
            self.inp = inp if self.inp is None else torch.cat([self.inp, inp], 1)

        # Step7: TODO prepare inputs for reprojection factor
        with torch.cuda.amp.autocast(enabled=False):
            target, _ = self.video.reproject(ii, jj)
            # Returns a tensor filled with the scalar value 0 , with the same size as input
            weight = torch.zeros_like(target)

        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)

        # Step8: constrcut reprojection factor for new edges
        self.net = net if self.net is None else torch.cat([self.net, net], 1)

        self.target = torch.cat([self.target, target], 1)
        self.weight = torch.cat([self.weight, weight], 1)

    @torch.cuda.amp.autocast(enabled=True)
    def rm_factors(self, mask, store=False):
        """ drop edges from factor graph """

        # store estimated factors
        if store:
            self.ii_inac = torch.cat([self.ii_inac, self.ii[mask]], 0)
            self.jj_inac = torch.cat([self.jj_inac, self.jj[mask]], 0)
            self.target_inac = torch.cat([self.target_inac, self.target[:,mask]], 1)
            self.weight_inac = torch.cat([self.weight_inac, self.weight[:,mask]], 1)

        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.age = self.age[~mask]
        
        if self.corr_impl == "volume":
            self.corr = self.corr[~mask]

        if self.net is not None:
            self.net = self.net[:,~mask]

        if self.inp is not None:
            self.inp = self.inp[:,~mask]

        self.target = self.target[:,~mask]
        self.weight = self.weight[:,~mask]


    @torch.cuda.amp.autocast(enabled=True)
    def rm_keyframe(self, ix):
        """ drop edges from factor graph """


        with self.video.get_lock():
            self.video.images[ix] = self.video.images[ix+1]
            self.video.poses[ix] = self.video.poses[ix+1]
            self.video.disps[ix] = self.video.disps[ix+1]
            self.video.disps_sens[ix] = self.video.disps_sens[ix+1]
            self.video.intrinsics[ix] = self.video.intrinsics[ix+1]

            self.video.nets[ix] = self.video.nets[ix+1]
            self.video.inps[ix] = self.video.inps[ix+1]
            self.video.fmaps[ix] = self.video.fmaps[ix+1]

        m = (self.ii_inac == ix) | (self.jj_inac == ix)
        self.ii_inac[self.ii_inac >= ix] -= 1
        self.jj_inac[self.jj_inac >= ix] -= 1

        if torch.any(m):
            self.ii_inac = self.ii_inac[~m]
            self.jj_inac = self.jj_inac[~m]
            self.target_inac = self.target_inac[:,~m]
            self.weight_inac = self.weight_inac[:,~m]

        m = (self.ii == ix) | (self.jj == ix)

        self.ii[self.ii >= ix] -= 1
        self.jj[self.jj >= ix] -= 1
        self.rm_factors(m, store=False)


    '''
        Update Operator
            Key contribution of DROID-SLAM, use update operator for RAFT inside
            Overall process similar to forward() in droid-net.py
            Difference: that only happens in training, this happens in inference
    '''
    @torch.cuda.amp.autocast(enabled=True)
    def update(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, motion_only=False):
        """ run update operator on factor graph """

        # During inference, feature and correlation map are extracted before

        # Step1: use projection-based matching and lookup operator to obtain initial correspondences &
        #        extract motion-based flow features
        with torch.cuda.amp.autocast(enabled=False):
            coords1, mask = self.video.reproject(self.ii, self.jj)
            motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
            motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)
        
        # Step2: use defined loopup operator on correlation pyramid
        corr = self.corr(coords1)

        # Step3: feed into GRU to get correction value for correspondence and confidence value
        # Inputs: context feature (self.net), hidden state (self.inp), correlation feature (corr), flow feature (motn)
        # Outputs: correction term for correspondence field (delta) and associated confidence map (weight)
        #          updated hidden state (self.net), pixelwise damping factor (damping), upsampling mask for inverse depth (upmask)
        self.net, delta, weight, damping, upmask = \
            self.update_op(self.net, self.inp, corr, motn, self.ii, self.jj)

        if t0 is None:
            t0 = max(1, self.ii.min().item()+1)

        # Step4: prepare inputs for BA layer
        with torch.cuda.amp.autocast(enabled=False):
            # Step4.1: get corrected corrspondence with initial correspondence and correction term
            self.target = coords1 + delta.to(dtype=torch.float)
            self.weight = weight.to(dtype=torch.float)

            ht, wd = self.coords0.shape[0:2]
            self.damping[torch.unique(self.ii)] = damping

            # Step4.2: TODO use_inactive?
            if use_inactive:
                m = (self.ii_inac >= t0 - 3) & (self.jj_inac >= t0 - 3)
                ii = torch.cat([self.ii_inac[m], self.ii], 0)
                jj = torch.cat([self.jj_inac[m], self.jj], 0)
                target = torch.cat([self.target_inac[:,m], self.target], 1)
                weight = torch.cat([self.weight_inac[:,m], self.weight], 1)

            else:
                ii, jj, target, weight = self.ii, self.jj, self.target, self.weight


            # Step4.3: prepare damping factor TODO what's EP?
            damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP

            target = target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
            weight = weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()

            # Step5: feed into DBA layer to obtain delta value for pose and inverse depth
            # Inputs: corrected correspondence field (target) and associated confidence map (weight), 
            #         pixelwise damping factor (damping), index pairs (ii, jj), timestamp pairs (t0, t1)
            #         iteration times (itrs), TODO lm, ep motion_only
            self.video.ba(target, weight, damping, ii, jj, t0, t1, 
                itrs=itrs, lm=1e-4, ep=0.1, motion_only=motion_only)
        
            # Step6: use unfold() to upsample estimated optical flow to match the input image size
            if self.upsample:
                self.video.upsample(torch.unique(self.ii), upmask)

        # Step7: count updated times
        self.age += 1


    '''
        Update operator
            Reduced memory implementation used in backend
    '''
    @torch.cuda.amp.autocast(enabled=False)
    def update_lowmem(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, steps=8):
        """ run update operator on factor graph - reduced memory implementation """

        # alternate corr implementation
        t = self.video.counter.value

        num, rig, ch, ht, wd = self.video.fmaps.shape
        corr_op = AltCorrBlock(self.video.fmaps.view(1, num*rig, ch, ht, wd))

        for step in range(steps):
            print("Global BA Iteration #{}".format(step+1))
            with torch.cuda.amp.autocast(enabled=False):
                coords1, mask = self.video.reproject(self.ii, self.jj)
                motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
                motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)

            s = 8
            for i in range(0, self.jj.max()+1, s):
                v = (self.ii >= i) & (self.ii < i + s)
                iis = self.ii[v]
                jjs = self.jj[v]

                ht, wd = self.coords0.shape[0:2]
                corr1 = corr_op(coords1[:,v], rig * iis, rig * jjs + (iis == jjs).long())

                with torch.cuda.amp.autocast(enabled=True):
                 
                    net, delta, weight, damping, upmask = \
                        self.update_op(self.net[:,v], self.video.inps[None,iis], corr1, motn[:,v], iis, jjs)

                    if self.upsample:
                        self.video.upsample(torch.unique(iis), upmask)

                self.net[:,v] = net
                self.target[:,v] = coords1[:,v] + delta.float()
                self.weight[:,v] = weight.float()
                self.damping[torch.unique(iis)] = damping

            damping = .2 * self.damping[torch.unique(self.ii)].contiguous() + EP
            target = self.target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
            weight = self.weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()

            # dense bundle adjustment
            self.video.ba(target, weight, damping, self.ii, self.jj, 1, t, 
                itrs=itrs, lm=1e-5, ep=1e-2, motion_only=False)

            self.video.dirty[:t] = True

    '''
        Covisibility graph
            Add time-interval-based neighbor frames to factor graph
    '''
    def add_neighborhood_factors(self, t0, t1, r=3):
        """ add edges between neighboring frames within radius r """

        # Step1: constructing 2D matrix to easily find nearby keyframes
        # ii - 2D tensor (t1-t0, t1-t0) - [t0,   t0, ..,   t0; t0+1, t0+1, ..., t0+1; t1-1, t1-1, .., t1-1]
        # jj - 2D tensor (t1-t0, t1-t0) - [t0, t0+1, .., t1-1;   t1, t1+1, ...,  t-1;   t1, t1+1, ..,  t-1]
        ii, jj = torch.meshgrid(torch.arange(t0,t1), torch.arange(t0,t1))

        # Step2: flatten the 2D matrix into 1D tensor
        # reshape - append the following lines into 1D tensor
        ii = ii.reshape(-1).to(dtype=torch.long, device=self.device)
        jj = jj.reshape(-1).to(dtype=torch.long, device=self.device)

        c = 1 if self.video.stereo else 0

        # Step3: select nearby keyframes which are within 3 timesteps apart
        keep = ((ii - jj).abs() > c) & ((ii - jj).abs() <= r)

        # Step4: add corresponding edges to factor graph
        self.add_factors(ii[keep], jj[keep]) # ii[keep] or jj[keep] - 1D tensor

    '''
        Covisibility graph
            Add distance-based proximity frames to factor graph
            non-max suppression (nms) TODO detailed implementation
    '''
    def add_proximity_factors(self, t0=0, t1=0, rad=2, nms=2, beta=0.25, thresh=16.0, remove=False):
        """ add edges to the factor graph based on distance """

        # Step1: constructing 2D matrix to easily find nearby keyframes TODO print value of t
        t = self.video.counter.value
        # In some cases, t0=t1=0, thus ix, jx and following ii, jj will be none TODO print shape
        ix = torch.arange(t0, t) # 1D tensor (t-t0+1) [t0, t0+1, ..., t-1]
        jx = torch.arange(t1, t) # 1D tensor (t-t1+1) [t1, t1+1, ..., t-1]

        # ii - 2D tensor (t-t0, t-t1) - [t0,   t0, ..,  t0; t0+1, t0+1, ..., t0+1; t-1,  t-1, .., t-1]
        # jj - 2D tensor (t-t1, t-t0) - [t1, t1+1, .., t-1;   t1, t1+1, ...,  t-1;  t1, t1+1, .., t-1]
        ii, jj = torch.meshgrid(ix, jx)

        # Step2: flatten the 2D matrix into 1D tensor
        ii = ii.reshape(-1) # 1D tensor
        jj = jj.reshape(-1) # 1D tensor

        # Step3: compute frame distance based on optical flow value
        # shape of d: (N, N) if ii=none else 1D tensor - ii.size(0)
        # For backend, d is 1D tensor;
        # For fronend, d is 1D tensor in initialization, and 2D tensor in tracking.
        d = self.video.distance(ii, jj, beta=beta)
        d[ii - rad < jj] = np.inf # TODO invalidify what?
        d[d > 100] = np.inf       # invalidify too large depth (TODO too close?)

        # Step4: suppress neighboring edges within a predefined distance threshold
        # TODO why include bad and inactive factors?
        ii1 = torch.cat([self.ii, self.ii_bad, self.ii_inac], 0) # concat using the first dim -> 1D tensor
        jj1 = torch.cat([self.jj, self.jj_bad, self.jj_inac], 0)
        # TODO understand detailed logic for this part
        for i, j in zip(ii1.cpu().numpy(), jj1.cpu().numpy()):
            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    # Chebyshev distance between index pairs: TODO ||(i,j) - (k,l)||∞ = max(|i-k|,|j-l|)
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            # TODO suppress what?
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf

        # Step5: add temporally adjacent keyframes to factor graph
        # TODO es - why es is empty if enable backend? detailed investigation
        es = [] # shape - [N, 2]
        for i in range(t0, t):
            if self.video.stereo:
                es.append((i, i))
                d[(i-t0)*(t-t1) + (i-t1)] = np.inf

            # TODO this part seems to append sth for sure?
            for j in range(max(i-rad-1,0), i):
                es.append((i,j))
                es.append((j,i))
                d[(i-t0)*(t-t1) + (j-t1)] = np.inf

        # Step6: sample new edges from the distance matrix in order of increasing flow and suppress neighboring edges
        # Returns the indices that sort a tensor along a given dimension in ascending order by value
        ix = torch.argsort(d)
        for k in ix:
            # TODO this threshold seems large
            if d[k].item() > thresh:
                continue

            # avoid inserting new factors to graph when exceeds maximum
            if len(es) > self.max_factors:
                break

            i = ii[k]
            j = jj[k]
            
            # TODO why bidirectional?
            es.append((i, j))
            es.append((j, i))

            # same block as before
            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    # Chebyshev distance between index pairs: TODO ||(i,j) - (k,l)||∞ = max(|i-k|,|j-l|)
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf

        # Step7: post-processing of index pairs and add corresponding edges to factor graph
        # TODO undestand why enable backend in a separate thread will fail
        #      es is empty if backend is enabled?
        # torch.as_tensor - converts data into a tensor
        # unbind - removes a tensor dimension (the last dim)
        # es - [[i1,j1], [j1,i1], .., [in,jn], [jn,in]] : 2D tensor [N, 2]
        # ii - [i1, j1, ..., in, jn]; jj - [j1, i1, ..., jn, in] : both 1D tensors
        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)
        self.add_factors(ii, jj, remove)
