import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class ProjectQaudMesh2Image(nn.Module):
  """Differentiable renderer for cuboids, implemented following SoftRas"""

  def __init__(self, K, texsize=7, near_plane=0., far_plane=None):
    """ Initialization
    Args:
        K (torch.FloatTensor): camera intrinsics
        texsize (int): texture resolution of each quadrilateral face
        near_plane (float): near plane for clipping the depth
        far_plane (float): far plane for clipping the depth
    """
    super(ProjectQaudMesh2Image, self).__init__()
    self.K = K
    self.im_width = int(K[0,2])*2
    self.im_height = int(K[1,2])*2
    
    self.near_plane = near_plane
    self.far_plane = far_plane

    # represent each quadrilateral face as a (dim_w)x(dim_h) grid
    self.dim_w = texsize 
    self.dim_h = texsize 

    # grid points on a single quadmesh, normalized to [-1.0, 1.0]
    ui, vi = np.meshgrid(range(self.dim_w), range(self.dim_h))
    ui = (ui.astype(np.float32) - (self.dim_w-1)/2) / ((self.dim_w-1)/2)
    vi = (vi.astype(np.float32) - (self.dim_h-1)/2) / ((self.dim_h-1)/2)
    depth = np.ones((self.dim_w*self.dim_h,1))
    vec = np.hstack((vi.reshape(-1,1), depth, ui.reshape(-1,1), depth)).astype(np.float32)
    vec = torch.tensor(vec).to(K.device)
    self.quad_pts = vec
    self.quad_corners = torch.tensor([[ 1., 0.,  1., 1.],
                                      [ 1., 0., -1., 1.],
                                      [-1., 0., -1., 1.],
                                      [-1., 0.,  1., 1.]], dtype=torch.float32).to(K.device)

    # camera rays
    x, y = np.meshgrid(range(self.im_width), range(self.im_height))
    x = x.flatten()
    y = y.flatten()
    w = np.ones_like(x)
    pixels = np.vstack((x+0.5, y+0.5, w)).transpose()
    pixels = torch.tensor(pixels).float().to(K.device)
    self.cam_rays = -(torch.inverse(K) @ pixels.t()).t()
    self.cam_rays = self.cam_rays.view(self.im_height, self.im_width, 3)

  def get_area(self, x):
    """ compute the area of the projected cuboids
    Args:
        x (torch.FloatTensor): vertices of the quadmesh, BxNx4x2
    Returns:
        area (torch.FloatTensor)
    """
    bs = x.shape[0]
    x = x.view(-1, 4, 2)
    areas = []
    neighbours = [[0,1], [1,2], [2,3], [3,0]]
    for i,j in neighbours:
        area = torch.abs( x[:,i,0]*x[:,j,1] - x[:,i,1]*x[:,j,0] ) 
        areas.append(area.view(-1,1))
    areas = torch.cat(areas, dim=1)
    areas = torch.mean(areas, dim=1)

    # normalize
    areas = areas/(self.im_width*self.im_height)

    # quadmesh area to cuboid area
    areas = torch.sum(areas.view(bs, -1), dim=1) / 2.0

    return areas

  def forward(self, RT, quad_RT, quad_feat, quad_scale, sigma=1e-5, gamma=1e-4):
    """ Project to image space
    Args:
        RT (torch.FloatTensor): camera extrinsics, Bx3x4
        quad_RT (torch.FloatTensor): pose of quadmeshes, BxNx4x4
        quad_feat (torch.FloatTensor): features of the quadmesh, BxNx(HxW)xC
        quad_scale (torch.FloatTensor): overall scale / scale in H and W, BxN / BxNx2
        sigma (float): sharpness of the soft projection
        gamma (float): transparency of the soft projection
    Returns:
        img (torch.FloatTensor): projected feature of the cuboids, BxCxHxW
        alpha (torch.FloatTensor): projected alpha of the cuboids, Bx(HxW)
        depth (torch.FloatTensor): projected depth of the cuboids, Bx(HxW)
        area (torch.FloatTensor): area of the projected cuboids, B
        center_depth (torch.FloatTensor): depth of the cuboid center, B
    """ 

    device = RT.device
    bs = quad_RT.shape[0]
    ns = quad_RT.shape[1]

    # if one scalar is provided for each quadmesh
    if len(quad_scale.shape)==2:
        quad_scale_pad = quad_scale.view(-1,-1,1).expand(-1,-1,4)
        quad_scale = quad_scale.view(-1,-1,1).expand(-1,-1,2)
    # if scale if provided for height and width seperately
    elif len(quad_scale.shape)==3:
        quad_scale_pad = torch.ones(quad_scale.shape[0], quad_scale.shape[1], 4).float().to(quad_scale.device)
        quad_scale_pad[:,:,0] = quad_scale[:,:,0]
        quad_scale_pad[:,:,2] = quad_scale[:,:,1]
    else:
        raise ValueError('Unsupported scale shape!')

    RT = torch.cat((RT, torch.tensor([0.,0.,0.,1.],dtype=torch.float32,device=RT.device).view(1,1,4).expand(bs, -1, -1)), dim=1)

    quad2cam = torch.matmul(RT.view(bs,1,4,4), quad_RT) # BxNx4x4
    pts_normal = quad2cam[:,:,:3,1]
    pts_location = quad2cam[:,:,:3,3]

    # apply scaling to quadmesh
    # BxNx4x4
    quad_corners = quad_scale_pad.view(bs,-1,1,4) * self.quad_corners.view(1,1,4,4).to(device)
    quad_corners[:,:,:,3] = 1

    # quadmesh to world coordinate
    # ((BxN)x1x4x4) x (Px4x1) --> ((BxN)xPx4x1)
    # P=(dim_w)x(dim_h), number of points
    pts_world = torch.matmul(quad_RT.view(bs*ns,1,4,4), self.quad_pts.view(-1, 4, 1).to(device))
    # ((BxN)x1x4x4) x ((BxN)xPx4x1) --> ((BxN)xPx4x1)
    corner_world = torch.matmul(quad_RT.view(bs*ns,1,4,4), quad_corners.view(bs*ns, -1, 4, 1))

    # world to camera coordinate
    # (Bx1x4x4) x (Bx(NxP)x4x1) --> (Bx(NxP)x4x1)
    pts_cam = torch.matmul(RT.view(bs,1,4,4), pts_world.view(bs,-1,4,1))
    pts_cam = pts_cam[:,:,0:3]
    corner_cam = torch.matmul(RT.view(bs,1,4,4), corner_world.view(bs,-1,4,1))
    corner_cam = corner_cam[:,:,0:3]
    # Bx(NxP)x3
    pts_cam = pts_cam.view(bs,-1,3)
    # BxNx4x3
    corner_cam = corner_cam.view(bs,ns,-1,3)
    
    # project onto image plane
    quad_feat_ = quad_feat.view(bs, ns, self.dim_h, self.dim_w, -1).permute(0,1,4,2,3) # BxNxhxwxC --> BxNxCxhxw
    img, alpha, depth = project_cuboids(pts_normal, corner_cam, self.cam_rays.to(device), quad_feat_, quad2cam, quad_scale, sigma, gamma,
                                                                         near_plane=self.near_plane, far_plane=self.far_plane)

    img = img.view(bs, -1, self.im_height, self.im_width) 
    alpha = alpha.view(bs, -1, self.im_height, self.im_width) 
    depth = depth.reshape(bs, 1, self.im_height, self.im_width)

    # computate the area of each projected quadmesh regardless if it is on the image plane
    corner_cam = corner_cam.view(bs*ns, -1, 3)
    corner_uv = (self.K.unsqueeze(0).to(device) @ corner_cam.transpose(1,2)).transpose(1,2)
    corner_uv = corner_uv / corner_uv[:,:,-1:] 
    area = self.get_area(corner_uv[:,:,0:2].view(bs, ns, -1, 2))

    corner_cam = corner_cam.view(bs, -1, 3)
    center_depth = torch.mean(corner_cam[:, :, -1], dim=1)
    
    return img, alpha, depth, area, center_depth

def dis2lineseg(p, cross, c0, c1):
  """Distance of a point to a line segment"""
  dot0 = torch.sum((p - c0) * (c1-c0), dim=2)
  dot1 = torch.sum((p - c1) * (c0-c1), dim=2)
  # distance to c0
  dis_c0 = torch.norm(p - c0, dim=2)
  # distance to c1
  dis_c1 = torch.norm(p - c1, dim=2)
  # distance to line
  dis = dis_c0*((dot0<0.0).type(torch.float32)) + dis_c1*((dot1<0.0).type(torch.float32))
  on_lineseg = (dot0>0) * (dot1>0)
  if on_lineseg.sum()>0:
    c10 = torch.norm(c1-c0, dim=2)+1e-3
    c10 = c10.expand_as(on_lineseg) # Nx1 -> Nx(HxW)
    dis[on_lineseg] = torch.norm(cross[on_lineseg], dim=1) / c10[on_lineseg]
  return dis

def project_faces(n, p0, c, l, featmap, quad2cam, scale, sigma, far_plane=None):
  """
  Project quadrilateral faces to the image plane
  Args:
      n (torch.FloatTensor): normal vector of the quadrilateral faces, Nx3
      p0 (torch.FloatTensor): center point of the quadrilateral faces, Nx3
      c (torch.FloatTensor): corners of the quadrilateral faces, Nx4x3
      l (torch.FloatTensor): camera rays, HxWx3 
      featmap (torch.FloatTensor): feature maps on the quadrilateral faces, NxCxHxW
      quad2cam (torch.FloatTensor): transformation matrix from quadmesh coordinate to cam coordinate, Nx4x4
      scale (torch.FloatTensor): scale of quadmesh 
      sigma (float): sharpness of the soft projection
      far_plane (float): far plane for clipping the depth
  Returns:
      image (torch.FloatTensor): projected feature of the quadmeshes, BxCxHxW
      alpha (torch.FloatTensor): projected alpha of the quadmeshes, Bx(HxW)
      depth (torch.FloatTensor): projected depth of the quadmeshes, Bx(HxW)
  """
  im_height = l.shape[0]
  im_width = l.shape[1]
  HW = im_height*im_width
  l = l.view(1, -1, 3) # 1x(HxW)x3
  n = n.view(-1, 1, 3) # Nx1x3
  p0 = p0.view(-1, 1, 3) # Nx1x3
  far_plane = 100. if not far_plane else far_plane

  # works in camera coordinate system, origin is camera center
  l0 = 0

  # 1. find out the intersected point of the rays and the plane
  # p is the intersected point, p = l0 + t*l
  #   dot(p - p0, n)=0
  #   dot(l0 + t*l - p0, n)=0
  #   dot(p0 - l0, n) - t*dot(l,n)=0
  demon = torch.sum(l*n, dim=2) # Nx(HxW)
  #mask_intersect = demon>1e-6
  t = torch.sum((p0 - l0) * n, dim=2) / (demon + 1e-6) # Nx(HxW)
  p = l0 + t.unsqueeze(2)*l # Nx(HxW)x3
  mask_front = t<0 # negative as normal vectors face towards camera

  # 2. if p is on the quadrilateral face 
  v = torch.abs(p - p0)
  cross0 = torch.cross((c[:,1:2,:]-c[:,0:1,:]).expand_as(v), p - c[:,0:1,:], dim=2)
  cross1 = torch.cross((c[:,2:3,:]-c[:,1:2,:]).expand_as(v), p - c[:,1:2,:], dim=2)
  cross2 = torch.cross((c[:,3:4,:]-c[:,2:3,:]).expand_as(v), p - c[:,2:3,:], dim=2)
  cross3 = torch.cross((c[:,0:1,:]-c[:,3:4,:]).expand_as(v), p - c[:,3:4,:], dim=2)
  dot0 = torch.sum(cross0 * n, dim=2)
  dot1 = torch.sum(cross1 * n, dim=2)
  dot2 = torch.sum(cross2 * n, dim=2)
  dot3 = torch.sum(cross3 * n, dim=2)
  mask_onboard = (dot0<0) * (dot1<0) * (dot2<0) * (dot3<0)

  # compute point to line segment distances
  dis0 = dis2lineseg(p, cross0, c[:,0:1,:], c[:,1:2,:]) #Nx(HxW)
  dis1 = dis2lineseg(p, cross1, c[:,1:2,:], c[:,2:3,:]) #Nx(HxW)
  dis2 = dis2lineseg(p, cross2, c[:,2:3,:], c[:,3:4,:]) #Nx(HxW)
  dis3 = dis2lineseg(p, cross3, c[:,3:4,:], c[:,0:1,:]) #Nx(HxW)
  dis,_ = torch.min(torch.cat((dis0.unsqueeze(0), dis1.unsqueeze(0), dis2.unsqueeze(0), dis3.unsqueeze(0)), dim=0), dim=0)
  dis = dis*dis
  mask_onboard = mask_onboard.float()
  dis = - far_plane*dis*mask_onboard + dis*(1-mask_onboard)
  dis = -dis
  alpha = torch.sigmoid(dis/sigma) 

  # filter out faces behind the camera
  alpha = alpha * mask_front.float()

  # transform p back to quadmesh coordinate
  cam2quad_R = quad2cam[:,:3,:3].transpose(1,2)
  cam2quad_T = -torch.matmul(cam2quad_R, quad2cam[:,:3,3].view(-1,3,1))
  p_quad = torch.matmul(cam2quad_R, p.permute(0,2,1)) + cam2quad_T #Nx3x3 * Nx3x(HxW) + Nx3x1

  # get uv coordinates in quadmesh coordinate
  # uv coordinates outsize of range [-1,1] will be padded with zero during grid_sample
  u = p_quad[:,0,:]/scale[:,0:1]  #Nx(HxW)
  v = p_quad[:,2,:]/scale[:,1:2]  #Nx(HxW)
  uv = torch.cat((u.unsqueeze(2),v.unsqueeze(2)), dim=2) # Nx(HxW)x2
  uv = uv.view(-1, im_height, im_width, 2)

  image = torch.nn.functional.grid_sample(featmap, uv, padding_mode='border')
  
  depth = p[:,:,2]
  mask_onboard = mask_onboard.type(torch.float32)
  depth = depth * mask_onboard +  1e+6 * (1-mask_onboard)

  return image, alpha, depth


def project_cuboids(normals, corners, rays, quad_feat, quad2cam, scale, sigma=1e-5, gamma=1e-4,
                                near_plane=0., far_plane=None):
  """
  Project cuboids onto the image plane following SoftRas, each cuboid is represented by 6 quadmeshes
  Args:
      normals (torch.FloatTensor): normal vector of the quadmeshes, BxNx3
      corners (torch.FloatTensor): vertices coordinates of the quadmeshes, BxNx4x3
      rays (torch.FloatTensor): camera rays, HxWx3 
      quad_feat (torch.FloatTensor): quadmesh features, BxNxCxhxw
      quad2cam (torch.FloatTensor): quadmesh coordinate to camera coordinate mat, BxNx4x4
      scale (torch.FloatTensor): scale of the quadmeshes, BxNx2
      sigma (float): sharpness of the soft projection
      gamma (float): transparency of the soft projection
      near_plane (float): near plane for clipping the depth
      far_plane (float): far plane for clipping the depth
  Returns:
      image (torch.FloatTensor): projected feature of the cuboids, BxCxHxW
      alpha (torch.FloatTensor): projected alpha of the cuboids, Bx(HxW)
      depth (torch.FloatTensor): projected depth of the cuboids, Bx(HxW)
  """

  bs = normals.shape[0]
  ns = normals.shape[1]
  # project quadmeshes
  locations = torch.mean(corners, dim=2)
  images, alphas, depths = project_faces(normals.view(bs*ns,-1), 
          locations.view(bs*ns,-1), corners.view(bs*ns,*corners.shape[2:]), rays, 
          quad_feat.view(bs*ns,*quad_feat.shape[2:]), quad2cam.view(bs*ns,4,4), 
          scale.view(bs*ns,-1), sigma, far_plane)

  depths = depths.view(bs, ns, -1) # (BxN)x(HxW) -> BxNx(HxW)
  images = images.view(bs, ns, *images.shape[1:]) # (BxN)xCxHxW -> BxNxCxHxW
  alphas = alphas.view(bs, ns, -1) # (BxN)x(HxW) -> BxNx(HxW)

  # composite 6 quadmeshes as a cuboid
  alpha = 1-alphas[:,0]
  for idx in range(1,ns):
    alpha = alpha*(1-alphas[:,idx])
  alpha = (1-alpha).view(bs,-1) #Bx(HxW)

  if far_plane is None:
    far_plane = depths.max()
  
  # softmax aggregation following SoftRas
  mask_near = depths>near_plane
  depths = depths.clamp(min=near_plane, max=far_plane)
  inv_depths = (far_plane-depths)/(far_plane-near_plane)
  w = alphas*torch.exp(inv_depths/gamma)
  w = w / (torch.sum(w, dim=1, keepdim=True) + 0.1)

  # clip near planes
  # we do not clip far plane as we want soft edges
  w = w*(mask_near.float()) 

  w = w.view(bs, ns, 1, *images.shape[3:])
  image = torch.sum(images * w, dim=1)
  depth, min_indice = torch.min(depths, dim=1) # Bx(HxW)

  return image, alpha, depth
