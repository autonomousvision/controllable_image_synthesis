import torch
from torch import nn
import numpy as np

class ProjectPoint2Image(nn.Module):
  """Differentiable renderer for point cloud"""
  
  def __init__(self, K, im_width, im_height, uv_only=False):
    super(ProjectPoint2Image, self).__init__()
    self.K = K
    self.im_width = im_width
    self.im_height = im_height
    
    ui, vi = np.meshgrid(range(im_width), range(im_height))
    grid = np.hstack((vi.reshape(-1,1), ui.reshape(-1,1))).astype(np.float32)
    self.grid = torch.tensor(grid).to(K.device)

    # params of gaussian kernel at every projected point, adapt wrt. intrinsics
    self.sigma = K[0,0].item()/16

    # if uv_only then return the re-projected uv coordinates, not the image
    self.uv_only = uv_only

  def forward(self, RT, pts_3d, pts_feat, pts_scale):
    """Project onto image
    Args:
        RT (torch.FloatTensor): camera extrinsics, Bx3x4
        pts_3d (torch.FloatTensor): point locations, BxNx3
        pts_feat (torch.FloatTensor): point features, BxNxC
        pts_scale (torch.FloatTensor): point scales, BxN
    Returns:
        img (torch.FloatTensor): projected image, BxCxHxW
    """
    device = RT.device
    bs = RT.shape[0]
    R = RT[:, :3,:3]
    T = RT[:, :3, 3]

    if pts_3d.shape[1]==1: # larger blob if there is single point
        self.sigma = self.K[0,0].item()/16.
    else:
        self.sigma = self.K[0,0].item()/32.

    # transform points from world coordinate to camera coordinate 
    points_local = (R @ pts_3d.transpose(1,2)).transpose(1,2) + T.view(bs,1,3)

    # perspective projection
    points_proj = self.K.unsqueeze(0).to(device) @ points_local.transpose(1,2) # Bx3xN
    points_mask = points_proj[:,2]>0.1 #BxN
    u = points_proj[:,0,:]/points_proj[:,2,:].clamp(min=0.1)
    v = points_proj[:,1,:]/points_proj[:,2,:].clamp(min=0.1)
    uv = torch.cat((v.reshape(bs,-1,1), u.reshape(bs,-1,1)),dim=2)

    if self.uv_only:
      uvz = torch.cat((uv, points_proj[:,2,:].reshape(bs,-1,1)),dim=2)
      return uvz

    # project points to image plance with soft weights
    # to differientiate to the geometry
    distance = uv.view(bs,-1,1,2) - self.grid.view(1,1,-1,2).expand(bs,-1,-1,-1).to(device)  # B x N x (HxW) x 2
    distance_sq = distance[...,0]**2 + distance[...,1]**2 # B x N x (HxW) 

    weight = torch.exp(-distance_sq / (pts_scale.view(bs,-1,1) * self.sigma * self.sigma))
    weight = weight * points_mask.view(bs,-1,1).float()

    # sum up features from all 3d points for each grid point
    img = pts_feat.transpose(1,2) @ weight #  (B x C x N) x (B x N x (HxW)) --> B x C x (HxW)
    img = img.view(bs, -1, self.im_height, self.im_width) 
    return img
