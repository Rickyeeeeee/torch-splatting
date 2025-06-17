import pdb
import torch
import torch.nn as nn
import math
from einops import reduce, rearrange

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def homogeneous(points):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)

def homogeneous_vec(vec):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([vec, torch.zeros_like(vec[..., :1])], dim=-1)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R



def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)



def build_covariance_3d(s, r):
    L = build_scaling_rotation(s, r)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance
    # symm = strip_symmetric(actual_covariance)
    # return symm

def build_covariance_3d_2dgs(s, r):
    L = build_scaling_rotation(torch.cat([s, torch.ones(s.shape[0], 1, device=s.device)], dim=1), r)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance


def build_covariance_2d(
    mean3d, cov3d, viewmatrix, fov_x, fov_y, focal_x, focal_y
):
    # The following models the steps outlined by equations 29
	# and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	# Additionally considers aspect / scaling of viewport.
	# Transposes used to account for row-/column-major conventions.
    tan_fovx = math.tan(fov_x * 0.5)
    tan_fovy = math.tan(fov_y * 0.5)
    t = (mean3d @ viewmatrix[:3,:3]) + viewmatrix[-1:,:3]

    # truncate the influences of gaussians far outside the frustum.
    tx = (t[..., 0] / t[..., 2]).clip(min=-tan_fovx*1.3, max=tan_fovx*1.3) * t[..., 2]
    ty = (t[..., 1] / t[..., 2]).clip(min=-tan_fovy*1.3, max=tan_fovy*1.3) * t[..., 2]
    tz = t[..., 2]

    # Eq.29 locally affine transform 
    # perspective transform is not affine so we approximate with first-order taylor expansion
    # notice that we multiply by the intrinsic so that the variance is at the sceen space
    J = torch.zeros(mean3d.shape[0], 3, 3).to(mean3d)
    J[..., 0, 0] = 1 / tz * focal_x
    J[..., 0, 2] = -tx / (tz * tz) * focal_x
    J[..., 1, 1] = 1 / tz * focal_y
    J[..., 1, 2] = -ty / (tz * tz) * focal_y
    # J[..., 2, 0] = tx / t.norm(dim=-1) # discard
    # J[..., 2, 1] = ty / t.norm(dim=-1) # discard
    # J[..., 2, 2] = tz / t.norm(dim=-1) # discard
    W = viewmatrix[:3,:3].T # transpose to correct viewmatrix
    cov2d = J @ W @ cov3d @ W.T @ J.permute(0,2,1)
    
    # add low pass filter here according to E.q. 32
    filter = torch.eye(2,2).to(cov2d) * 0.3
    return cov2d[:, :2, :2] + filter[None]

def build_transforms(means3D, scales, rotations, viewmatrix, projmatrix):
    transforms = build_covariance_3d_2dgs(scales, rotations)
    p_view = means3D @ viewmatrix[:3,:3] + viewmatrix[-1:,:3]
    uv_view = transforms @ viewmatrix[:3,:3]
    M = torch.cat(
        [
            homogeneous_vec(uv_view[:,:2,:]),
            homogeneous(p_view.unsqueeze(1))
        ],
        dim=1) # M
    T = M @ projmatrix
    return T

def projection_ndc(points, viewmatrix, projmatrix):
    points_o = homogeneous(points) # object space
    points_h = points_o @ viewmatrix @ projmatrix # screen space # RHS
    p_w = 1.0 / (points_h[..., -1:] + 0.000001)
    p_proj = points_h * p_w
    p_view = points_o @ viewmatrix
    in_mask = p_view[..., 2] >= 0.2
    return p_proj, p_view, in_mask


@torch.no_grad()
def get_radius_old(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

@torch.no_grad()
def get_radius(transforms, cutoff=3.0, filterSize=0.707106):
    """
    Args:
        transforms: Tensor of shape (n, 3, 3)
        cutoff: Scalar cutoff value
        FilterSize: Scalar value to scale the cutoff

    Returns:
        radius: Tensor of shape (n,) with computed radius
        point_image: Tensor of shape (n, 2) with computed center
    """
    t = torch.tensor([cutoff**2, cutoff**2, -1.0], device=transforms.device)
    T0 = transforms[:, 0, :]  # shape: (n, 3)
    T1 = transforms[:, 1, :]
    T2 = transforms[:, 2, :]

    d = torch.sum(t * (T2 * T2), dim=1)  # shape: (n,)
    valid = d != 0

    radius = torch.zeros(transforms.shape[0], device=transforms.device)
    point_image = torch.zeros((transforms.shape[0], 2), device=transforms.device)

    if valid.any():
        f = (1.0 / d[valid]).unsqueeze(1) * t  # shape: (nv, 3)
        p_x = torch.sum(f * (T0[valid] * T2[valid]), dim=1)
        p_y = torch.sum(f * (T1[valid] * T2[valid]), dim=1)
        p = torch.stack([p_x, p_y], dim=1)

        dot00 = torch.sum(f * (T0[valid] * T0[valid]), dim=1)
        dot11 = torch.sum(f * (T1[valid] * T1[valid]), dim=1)

        h0_x = p_x * p_x - dot00
        h0_y = p_y * p_y - dot11
        h0 = torch.stack([h0_x, h0_y], dim=1)

        h = torch.sqrt(torch.clamp(h0, min=1e-4))
        extent = h

        r = torch.ceil(torch.clamp(
            torch.max(extent[:, 0], extent[:, 1]),
            min=cutoff * filterSize
        ))

        radius[valid] = r
        point_image[valid] = p

    return radius, point_image

@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max


from .utils.sh_utils import eval_sh
import torch.autograd.profiler as profiler
USE_PROFILE = False
import contextlib

class Gauss2DRenderer(nn.Module):
    """
    A gaussian splatting renderer

    >>> gaussModel = GaussModel.create_from_pcd(pts)
    >>> gaussRender = GaussRenderer()
    >>> out = gaussRender(pc=gaussModel, camera=camera)
    """

    def __init__(self, active_sh_degree=3, white_bkgd=True, **kwargs):
        super(Gauss2DRenderer, self).__init__()
        self.active_sh_degree = active_sh_degree
        self.debug = False
        self.white_bkgd = white_bkgd
        self.pix_coord = torch.stack(torch.meshgrid(torch.arange(256), torch.arange(256), indexing='xy'), dim=-1).to('cuda')
        self.filterSize = 0.707106
        
    
    def build_color(self, means3D, shs, camera):
        rays_o = camera.camera_center
        rays_d = means3D - rays_o
        color = eval_sh(self.active_sh_degree, shs.permute(0,2,1), rays_d)
        color = (color + 0.5).clip(min=0.0)
        return color
    
    def render_old(self, camera, means2D, cov2d, color, opacity, depths):
        radii = get_radius_old(cov2d)
        rect = get_rect(means2D, radii, width=camera.image_width, height=camera.image_height)
        
        self.render_color = torch.ones(*self.pix_coord.shape[:2], 3).to('cuda')
        self.render_depth = torch.zeros(*self.pix_coord.shape[:2], 1).to('cuda')
        self.render_alpha = torch.zeros(*self.pix_coord.shape[:2], 1).to('cuda')

        TILE_SIZE = 64
        for h in range(0, camera.image_height, TILE_SIZE):
            for w in range(0, camera.image_width, TILE_SIZE):
                # check if the rectangle penetrate the tile
                over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
                over_br = rect[1][..., 0].clip(max=w+TILE_SIZE-1), rect[1][..., 1].clip(max=h+TILE_SIZE-1)
                in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) # 3D gaussian in the tile 
                
                if not in_mask.sum() > 0:
                    continue

                P = in_mask.sum()
                tile_coord = self.pix_coord[h:h+TILE_SIZE, w:w+TILE_SIZE].flatten(0,-2)
                sorted_depths, index = torch.sort(depths[in_mask])
                sorted_means2D = means2D[in_mask][index]
                sorted_cov2d = cov2d[in_mask][index] # P 2 2
                sorted_conic = sorted_cov2d.inverse() # inverse of variance
                sorted_opacity = opacity[in_mask][index]
                sorted_color = color[in_mask][index]
                dx = (tile_coord[:,None,:] - sorted_means2D[None,:]) # B P 2
                
                gauss_weight = torch.exp(-0.5 * (
                    dx[:, :, 0]**2 * sorted_conic[:, 0, 0] 
                    + dx[:, :, 1]**2 * sorted_conic[:, 1, 1]
                    + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 0, 1]
                    + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1, 0]))
                
                alpha = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.99) # B P 1
                T = torch.cat([torch.ones_like(alpha[:,:1]), 1-alpha[:,:-1]], dim=1).cumprod(dim=1)
                acc_alpha = (alpha * T).sum(dim=1)
                tile_color = (T * alpha * sorted_color[None]).sum(dim=1) + (1-acc_alpha) * (1 if self.white_bkgd else 0)
                tile_depth = ((T * alpha) * sorted_depths[None,:,None]).sum(dim=1)
                self.render_color[h:h+TILE_SIZE, w:w+TILE_SIZE] = tile_color.reshape(TILE_SIZE, TILE_SIZE, -1)
                self.render_depth[h:h+TILE_SIZE, w:w+TILE_SIZE] = tile_depth.reshape(TILE_SIZE, TILE_SIZE, -1)
                self.render_alpha[h:h+TILE_SIZE, w:w+TILE_SIZE] = acc_alpha.reshape(TILE_SIZE, TILE_SIZE, -1)

        return {
            "render": self.render_color,
            "depth": self.render_depth,
            "alpha": self.render_alpha,
            "visiility_filter": radii > 0,
            "radii": radii
        }

    def render(self, camera, means2D, transforms, color, opacity, depths):
        radii, point_image = get_radius(transforms[..., :3], filterSize=self.filterSize) # Fix this
        rect = get_rect(point_image, radii, width=camera.image_width, height=camera.image_height)

        self.render_color = torch.ones(*self.pix_coord.shape[:2], 3).to('cuda')
        self.render_depth = torch.zeros(*self.pix_coord.shape[:2], 1).to('cuda')
        self.render_alpha = torch.zeros(*self.pix_coord.shape[:2], 1).to('cuda')

        TILE_SIZE = 64
        for h in range(0, camera.image_height, TILE_SIZE):
            for w in range(0, camera.image_width, TILE_SIZE):
                # check if the rectangle penetrate the tile
                over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
                over_br = rect[1][..., 0].clip(max=w+TILE_SIZE-1), rect[1][..., 1].clip(max=h+TILE_SIZE-1)
                in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1])

                if not in_mask.sum() > 0:
                    continue

                P = in_mask.sum()
                tile_coord = self.pix_coord[h:h+TILE_SIZE, w:w+TILE_SIZE].flatten(0,-2)
                sorted_depths, index = torch.sort(depths[in_mask])
                sorted_means2D = means2D[in_mask][index]
                sorted_transforms = transforms[in_mask][index] # P 2 2
                sorted_opacity = opacity[in_mask][index]
                sorted_color = color[in_mask][index]

                # Compute Gaussian weights (gauss_weight) for each pixel in tile_coord and each Gaussian

                # Prepare components
                Tu = sorted_transforms[..., 0]  # shape: (P, 3)
                Tv = sorted_transforms[..., 1]
                Tw = sorted_transforms[..., 3]

                pix = tile_coord  # shape: (B, 2)
                P = sorted_transforms.shape[0]
                B = pix.shape[0]

                # Project each pixel using Tu, Tv, Tw and compute p = k x l (cross product)
                pix_x = pix[:, 0]  # shape: (B, 1)
                pix_y = pix[:, 1]

                # Compute planes
                Tu = rearrange(Tu, 'P u -> 1 P u')  # shape: (P, 1, 3)
                Tv = rearrange(Tv, 'P v -> 1 P v')  # shape: (P, 1, 3)
                Tw = rearrange(Tw, 'P w -> 1 P w')  # shape: (P, 1, 3)
                pix_x = rearrange(pix_x, 'B -> B 1 1')  # shape: (B, 1, 1)
                pix_y = rearrange(pix_y, 'B -> B 1 1')  # shape: (B, 1, 1)
                k = pix_x * Tw - Tu  # shape: (B, P, 3)
                l = pix_y * Tw - Tv  # shape: (B, P, 3)

                # Cross product (k x l)
                px = k[...,1]*l[...,2] - k[...,2]*l[...,1]
                py = k[...,2]*l[...,0] - k[...,0]*l[...,2]
                pz = k[...,0]*l[...,1] - k[...,1]*l[...,0]
                p = torch.stack([px, py, pz], dim=-1)  # shape: (B, P, 3)

                # Avoid division by zero
                mask = p[..., 2] != 0
                p[..., 2] = torch.where(mask, p[..., 2], torch.ones_like(p[..., 2]))

                # s = p.xy / p.z
                s = p[..., :2] / p[..., 2:3]  # shape: (B, P, 2)
                rho3d = (s**2).sum(dim=-1)  # shape: (B, P)

                # 2D distance term
                xy = sorted_means2D  # shape: (P, 2)
                pixf = pix.unsqueeze(1)  # shape: (B, 1, 2)
                d = xy.unsqueeze(0) - pixf  # shape: (B, P, 2)
                rho2d = (d**2).sum(dim=-1) * (1.0 / (self.filterSize**2))

                # Combine them
                rho = torch.minimum(rho3d, rho2d)  # shape: (B, P)
                gauss_weight = torch.exp(-0.5 * rho)  # shape: (B, P)

                # Not the same as the original code
                alpha = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.99) # B P 1
                T = torch.cat([torch.ones_like(alpha[:,:1]), 1-alpha[:,:-1]], dim=1).cumprod(dim=1)
                acc_alpha = (alpha * T).sum(dim=1)
                tile_color = (T * alpha * sorted_color[None]).sum(dim=1) + (1-acc_alpha) * (1 if self.white_bkgd else 0)
                tile_depth = ((T * alpha) * sorted_depths[None,:,None]).sum(dim=1)
                self.render_color[h:h+TILE_SIZE, w:w+TILE_SIZE] = tile_color.reshape(TILE_SIZE, TILE_SIZE, -1)
                self.render_depth[h:h+TILE_SIZE, w:w+TILE_SIZE] = tile_depth.reshape(TILE_SIZE, TILE_SIZE, -1)
                self.render_alpha[h:h+TILE_SIZE, w:w+TILE_SIZE] = acc_alpha.reshape(TILE_SIZE, TILE_SIZE, -1)

        return {
            "render": self.render_color,
            "depth": self.render_depth,
            "alpha": self.render_alpha,
            "visiility_filter": radii > 0,
            "radii": radii
        }

    def forward(self, camera, pc, **kwargs):
        means3D = pc.get_xyz
        opacity = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation
        shs = pc.get_features
        
        if USE_PROFILE:
            prof = profiler.record_function
        else:
            prof = contextlib.nullcontext
            
        with prof("projection"):
            mean_ndc, mean_view, in_mask = projection_ndc(means3D, 
                    viewmatrix=camera.world_view_transform, 
                    projmatrix=camera.projection_matrix)
            mean_ndc = mean_ndc[in_mask]
            mean_view = mean_view[in_mask]
            depths = mean_view[:,2]
        
        with prof("build color"):
            color = self.build_color(means3D=means3D, shs=shs, camera=camera)
        
        with prof("build cov3d"):
            T = build_transforms(
                means3D=means3D[in_mask], 
                scales=scales[in_mask], 
                rotations=rotations[in_mask],
                viewmatrix=camera.world_view_transform, 
                projmatrix=camera.projection_matrix
            )
            
        # with prof("build cov2d"):
        #     cov2d = build_covariance_2d(
        #         mean3d=means3D, 
        #         cov3d=cov3d, 
        #         viewmatrix=camera.world_view_transform,
        #         fov_x=camera.FoVx, 
        #         fov_y=camera.FoVy, 
        #         focal_x=camera.focal_x, 
        #         focal_y=camera.focal_y)

        #     mean_coord_x = ((mean_ndc[..., 0] + 1) * camera.image_width - 1.0) * 0.5
        #     mean_coord_y = ((mean_ndc[..., 1] + 1) * camera.image_height - 1.0) * 0.5
        #     means2D = torch.stack([mean_coord_x, mean_coord_y], dim=-1)
        
        # with prof("render"):
        #     rets = self.render_old(
        #         camera = camera, 
        #         means2D=means2D,
        #         cov2d=cov2d,
        #         color=color,
        #         opacity=opacity, 
        #         depths=depths,
        #     )

        with prof("render"):
            transforms = T
            means2D = mean_ndc[:, :2]
            rets = self.render(
                camera=camera, 
                means2D=means2D,
                transforms=transforms,
                color=color,
                opacity=opacity, 
                depths=depths,
            )

        return rets
