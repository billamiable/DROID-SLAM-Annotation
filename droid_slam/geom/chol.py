import torch
import torch.nn.functional as F
import geom.projective_ops as pops

'''
    BA Solver
        CholeskySolver, similar to the one in g2o
'''
class CholeskySolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b):
        # don't crash training if cholesky decomp fails
        try:
            U = torch.linalg.cholesky(H)
            xs = torch.cholesky_solve(b, U)
            ctx.save_for_backward(U, xs)
            ctx.failed = False
        except Exception as e:
            print(e)
            ctx.failed = True
            xs = torch.zeros_like(b)

        return xs

    @staticmethod
    def backward(ctx, grad_x):
        if ctx.failed:
            return None, None

        U, xs = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x, U)
        dH = -torch.matmul(xs, dz.transpose(-1,-2))

        return dH, dz

def block_solve(H, b, ep=0.1, lm=0.0001):
    """ solve normal equations """
    B, N, _, D, _ = H.shape
    I = torch.eye(D).to(H.device)
    H = H + (ep + lm*H) * I

    H = H.permute(0,1,3,2,4)
    H = H.reshape(B, N*D, N*D)
    b = b.reshape(B, N*D, 1)

    x = CholeskySolver.apply(H,b)
    return x.reshape(B, N, D)


def schur_solve(H, E, C, v, w, ep=0.1, lm=0.0001, sless=False):
    """ solve using shur complement """
    
    # Step1: preprocessing and prepare inputs
    B, P, M, D, HW = E.shape
    H = H.permute(0,1,3,2,4).reshape(B, P*D, P*D)
    E = E.permute(0,1,3,2,4).reshape(B, P*D, M*HW)
    # C^-1 = 1 / C (diagonal)
    Q = (1.0 / C).view(B, M*HW, 1) # C has damping factor included

    # Step2: TODO add pixelwise damping factor to the depth block? or similar to Laplace term?
    I = torch.eye(P*D).to(H.device)
    H = H + (ep + lm*H) * I
    
    v = v.reshape(B, P*D, 1)
    w = w.reshape(B, M*HW, 1)

    # Step3: prepare inputs for schur complement
    # E^T
    Et = E.transpose(1,2)
    # B - E * C^-1 * E^T
    S = H - torch.matmul(E, Q*Et)
    # v - E * C^-1 * w
    v = v - torch.matmul(E, Q*w)

    # Step4: substitute delta_depth and first obtain estimation for delta_pose
    dx = CholeskySolver.apply(S, v)
    if sless:
        return dx.reshape(B, P, D)

    # Step5: use delta_pose to derive delta_depth
    # delta_depth = C^-1 * (w - E^T * delta_pose)
    dz = Q * (w - Et @ dx)

    # Step6: reshape output delta variables
    dx = dx.reshape(B, P, D)
    dz = dz.reshape(B, M, HW)

    return dx, dz