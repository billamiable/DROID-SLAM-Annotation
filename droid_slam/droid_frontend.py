import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph


class DroidFrontend:
    def __init__(self, net, video, args):
        self.video = video
        self.update_op = net.update
        '''
            Local Bundle Adjustment
                Factor graph for local BA
        '''
        self.graph = FactorGraph(video, net.update, max_factors=48, upsample=args.upsample)

        # local optimization window
        self.t0 = 0
        self.t1 = 0

        # frontent variables
        self.is_initialized = False
        self.count = 0

        self.max_age = 25
        self.iters1 = 4
        self.iters2 = 2

        self.warmup = args.warmup
        self.beta = args.beta
        self.frontend_nms = args.frontend_nms
        self.keyframe_thresh = args.keyframe_thresh
        self.frontend_window = args.frontend_window
        self.frontend_thresh = args.frontend_thresh
        self.frontend_radius = args.frontend_radius

    '''
        Tracking
            Frontend steps other than initialization.
    '''
    def __update(self):
        """ add edges, perform update """

        self.count += 1
        self.t1 += 1

        # Step1: TODO remove edges from factor graph
        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        # Step2: add distance-based proximity edge to factor graph
        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0), 
            rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh, beta=self.beta, remove=True)

        # Step3: TODO provide initial value for disparity
        self.video.disps[self.t1-1] = torch.where(self.video.disps_sens[self.t1-1] > 0, 
           self.video.disps_sens[self.t1-1], self.video.disps[self.t1-1])

        # Step4: run multiple times of factor graph update
        for itr in range(self.iters1):
            self.graph.update(None, None, use_inactive=True)

        '''
            Initial Pose Estimation
                TODO Directly use pose from last frame or apply motion model?
        '''
        # Step5: set initial pose for next frame
        poses = SE3(self.video.poses)

        # Step6: remove redundant keyframes according to distance or update graph
        d = self.video.distance([self.t1-3], [self.t1-2], beta=self.beta, bidirectional=True)

        '''
            Remove redundant keyframes
                Remove according to distance
                TODO: which distance?
        '''
        if d.item() < self.keyframe_thresh:
            self.graph.rm_keyframe(self.t1 - 2)
            
            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1

        # Step7: re-run multiple times of factor graph update
        else:
            for itr in range(self.iters2):
                self.graph.update(None, None, use_inactive=True)

        # Step8: set pose for next iteration
        self.video.poses[self.t1] = self.video.poses[self.t1-1]
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()

        # Step9: update visualization
        self.video.dirty[self.graph.ii.min():self.t1] = True

    '''
        Initialization
            First stage when the SLAM starts.
    '''
    def __initialize(self):
        """ initialize the SLAM system """

        self.t0 = 0
        self.t1 = self.video.counter.value

        # Step1: add time-interval-based nearby edge into factor graph
        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)

        # Step2: run multiple times of graph update
        for itr in range(8):
            # only run one time for each iteration
            # TODO use_inactive
            self.graph.update(1, use_inactive=True)

        # Step3: add distance-based proximity edge into factor graph
        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)

        # Step4: run multiple times of graph update again
        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        # Step5: provide initial value for pose and disparity
        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()

        # Step6: post-processing after initialization
        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1-1].clone()
        self.last_disp = self.video.disps[self.t1-1].clone()
        self.last_time = self.video.tstamp[self.t1-1].clone()

        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[:self.t1] = True

        # Step7: TODO remove edges from factor graph
        self.graph.rm_factors(self.graph.ii < self.warmup-4, store=True)

    # use __call__ func when the class obeject is called - self.frontend()
    def __call__(self):
        """ main update """

        # Step1: do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize()
            
        # Step2: do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self.__update()

        
