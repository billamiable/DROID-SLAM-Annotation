import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph


class DroidBackend:
    def __init__(self, net, video, args):
        self.video = video
        self.update_op = net.update

        # global optimization window
        self.t0 = 0
        self.t1 = 0

        self.upsample = args.upsample
        self.beta = args.beta
        self.backend_thresh = args.backend_thresh
        self.backend_radius = args.backend_radius
        self.backend_nms = args.backend_nms
        
    @torch.no_grad()
    def __call__(self, steps=12):
        """ main update """

        t = self.video.counter.value
        # Step1: TODO normalize depth and poses
        if not self.video.stereo and not torch.any(self.video.disps_sens):
             self.video.normalize()

        '''
            Global Bundle Adjustment
                Still using factor graph for global BA
        '''
        # Step2: initialize factor graph for global BA
        graph = FactorGraph(self.video, self.update_op, corr_impl="alt", max_factors=16*t, upsample=self.upsample)

        # Step3: add distance-based proximity edge to factor graph
        graph.add_proximity_factors(rad=self.backend_radius, 
                                    nms=self.backend_nms, 
                                    thresh=self.backend_thresh, 
                                    beta=self.beta)

        # Step4: TODO update graph using reduced memory implementation
        graph.update_lowmem(steps=steps)

        # Step5: post-processing after global optimization
        graph.clear_edges()
        self.video.dirty[:t] = True
