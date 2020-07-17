import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from utils.commons import get_cube_RT
import neural_renderer as nr
import controllable_gan.render as dr
import os


class Renderer(nn.Module):
  """Renderer class with basic camera settings and functions"""
  def __init__(self, img_size=64, near_plane=0., far_plane=100.):
    super(Renderer, self).__init__()
    
    self.near_plane = near_plane
    self.far_plane = far_plane
    
    K = np.array([[img_size, 0.,  img_size//2],
                  [ 0., img_size, img_size//2],
                  [ 0., 0.,   1.]])
    K = torch.from_numpy(K).type(torch.float32)
    im_width = im_height = torch.tensor(img_size).type(torch.long)

    self.K = K
    self.im_width = im_width
    self.im_height = im_height
    
    self.RT = torch.zeros(3,4).float()
    self.RT[:3,:3] = torch.from_numpy(np.eye(3)).float()
    self.RT[2,3]=3

    self.sigma = 1e-2
    self.gamma = 1e-1
    self.constant = torch.tensor([0.,0.,0.,1.]).float()

  def get_pose_world(self, x_3d):
    r_mat = x_3d['R'].view(-1, 3, 3)
    t_vec = x_3d['t'].view(-1, 3, 1)

    tr = torch.cat((r_mat, t_vec), dim=2)
    cuboid_RT = torch.cat((tr, self.constant.view(-1,1,4).expand(tr.shape[0],-1,-1).to(r_mat.device)), dim=1)
    return cuboid_RT

  def get_pose_cam(self, x_3d, RT):
    cuboid_RT = self.get_pose_world(x_3d)
    cuboid_RT = RT @ cuboid_RT

    return cuboid_RT

  def transform(self, x_3d, vs):
    r_mat = x_3d['R']
    t_vec = x_3d['t']
    scale_vec = x_3d['scale']

    vs = vs * scale_vec.unsqueeze(1)
    vs = (r_mat @ vs.transpose(1,2)).transpose(1,2) + t_vec.unsqueeze(1)

    return vs


class RendererMesh(Renderer):
  """Mesh renderer for "cuboid" and "sphere" """
  def __init__(self, mesh=None, img_size=64, texture_channel=3, template_dir=None, background_color=[0,0,0], near_plane=0., far_plane=100.):
    """Mesh renderer initialization
    
    Args:
        mesh (Primitive, optional): Primitive template containing vertices and faces. Defaults to None.
        texture_channel (int, optional): Channel of texture map. Defaults to 3.
        template_dir (str, optional): Path to the template mesh, only used when the input mesh is None. Defaults to None.
        background_color (list, optional): Background color of the renderer. Defaults to [0,0,0].
    """
    super(RendererMesh, self).__init__(img_size=img_size, near_plane=near_plane, far_plane=far_plane)

    if mesh != None:
      v = mesh.v
      f = mesh.f
      texture_channel = mesh.n_channel
    else:
      assert template_dir != None and os.path.isfile(template_dir)
      v, f = nr.load_obj(template_dir)

    self.v = v[None] * 1.5
    self.f = f[None]
    self.texture_channel=texture_channel

    # neural mesh renderer
    self.renderer = nr.Renderer(image_size=64,  camera_mode='projection', background_color=background_color,
                                K = self.K[None], fill_back=True, orig_size=64, texture_channel=texture_channel,
                                near=self.near_plane, far=self.far_plane)
    # Make light only ambient.
    self.renderer.light_intensity_ambient = 1
    self.renderer.light_intensity_directional = 0

    self.n_v = self.v.shape[1]
    self.n_f = self.f.shape[1]

    self.tex_size = 4
    self.pred_uv = True


  def forward(self, x_3d, RT):
    """Render 3D meshes to 2D images
    
    Args:
        x_3d (dict): 3D primitives including poses parameters and features 
        RT (torch.FloatTensor): Camera poses 
    
    Returns:
        images (torch.FloatTensor): Rendered image, alpha and depth, NxCxHxW
        area (torch.FloatTensor): Mean area of projected mesh, N
        center_depth (torch.FloatTensor): depth of the center coordinate of the mesh, N
        cuboid_RT (torch.FloatTensor): The poses of the mesh in camera coordinate
    """

    device = x_3d['R'].device
    N = x_3d['R'].shape[0]

    assert RT.shape[0] == N
    RT = RT.to(device)

    texture = x_3d['feature']

    vs = self.transform(x_3d, self.v.to(device))
    fs = self.f.repeat(N, 1,1).to(device)
    ts = texture.unsqueeze(4).repeat(1,1,1,1,self.tex_size,1).to(device)

    # render mesh
    bg_depth = self.far_plane
    image,depth,alpha = self.renderer(vs, fs, ts, R=RT[:,:3,:3], t=RT[:,:3,3])
    depth = depth.clamp(0, bg_depth)
    images = torch.cat((image, alpha.unsqueeze(1), depth.unsqueeze(1)), dim=1)

    # compute depth at the center coordinate of the mesh
    t_vec = x_3d['t']
    center_depth = self.K.unsqueeze(0).to(device) @((RT[:,:3,:3]@t_vec.view(-1, 3, 1)) + RT[:,:,-1:])
    center_depth = center_depth[:,2].view(-1)

    # compute projected area
    area = alpha.view(N, -1).mean(dim=1)
    
    cuboid_RT = self.get_pose_cam(x_3d, RT)

    return images, area, center_depth, cuboid_RT


class RendererBG(RendererMesh):
  """Background renderer based on RendererMesh"""
  def __init__(self, bg_radius=2.0, texture_channel=3, template_dir='controllable_gan/templates/sphere_642.obj', near_plane=0., far_plane=100.):
    super(RendererBG, self).__init__(texture_channel=texture_channel, template_dir=template_dir, near_plane=near_plane, far_plane=far_plane)

    self.v = self.v / self.v.squeeze(0).norm(p=2, dim=1).max()    # normalize vertices to 1
    self.v = self.v * bg_radius # rescale the vertices to desired size
    self.pred_uv = True
    self.tex_size = 4

  def forward(self, texture, RT=None):
    device = texture.device
    batch_size = texture.shape[0]

    if RT is None:
        RT = self.RT.unsqueeze(0).repeat(batch_size, 1,1)
    else:
        assert RT.shape[0]==batch_size

    RT = RT.to(device)
    vs = self.v.repeat(batch_size, 1, 1).to(device)
    fs = self.f.repeat(batch_size, 1,1).to(device)
    ts = texture.unsqueeze(4).repeat(1,1,1,1,self.tex_size,1).to(device)

    image,depth,alpha = self.renderer(vs, fs, ts, R=RT[:,:3,:3], t=RT[:,:3,3].unsqueeze(1))
    image = torch.cat((image, alpha.unsqueeze(1), depth.unsqueeze(1)), dim=1)

    return image


class RendererQuadMesh(Renderer):
  """Mesh renderer for 'cuboid_sr', render quadrilateral meshes following SoftRas"""
  def __init__(self, cuboid, img_size=64, near_plane=0., far_plane=100.):
    """Initialization
    
    Args:
        mesh (Primitive): Cuboid primitive class with basic configurations (texsize, n_channel) 
    """
    super(RendererQuadMesh, self).__init__(img_size=img_size, near_plane=near_plane, far_plane=far_plane)

    # number of vertices on each dimension of the quadmesh
    # if 1 then apply flat color
    # if >1 then learn (texsize)x(texsize) feature for each face of the cuboid
    self.texsize = texsize = cuboid.texsize
    texsize = max(texsize, 2)

    self.quad_channel = cuboid.n_channel
    
    self.project = dr.ProjectQaudMesh2Image(self.K, texsize=texsize,
                                            near_plane=self.near_plane, 
                                            far_plane=self.far_plane)
    
    # initialize the faces with the same color
    self.quad_feat = torch.ones(6,texsize**2, self.quad_channel).float()

    self.cuboid_edge = torch.tensor([[0,1],[0,1],
                                     [0,2],[0,2],
                                     [1,2],[1,2]]).long()

  def get_cuboids(self, x_3d):
    """Convert 3D params to cuboids
    
    Args:
        x_3d (dict): 3D primitives including poses parameters and features 
    
    Returns:
        quad_RT (torch.FloatTensor): Pose of the 6 faces of the cuboids in world coordinate, (BxN)x6x4x4 
        quad_feat (torch.FloatTensor): Feature maps on the 6 faces of the cuboids, (BxN)x6x(DxD)xC
        quad_scale (torch.FloatTensor): Scale of the cuboids
        cuboid_RT (torch.FloatTensor): Pose of the cuboids in world coorindate, (BxN)x4x4
    """
    
    device = x_3d['R'].device
    N = x_3d['R'].shape[0]

    scale_vec = x_3d['scale']
    feat_vec = x_3d['feature']

    cuboid_RT = self.get_pose_world(x_3d) 

    quad_RT = get_cube_RT(scale_vec)
    quad_RT = cuboid_RT.view(N, 1, 4, 4) @ quad_RT # (BxN)x1x4x4 x (BxN)x6x4x4

    feat_vec = feat_vec.view(N, -1, self.texsize**2, self.quad_channel)
    # 1x6x4x3 x (BxN)x1x1x3 or (BxN)x6x1x3
    quad_feat = self.quad_feat.to(device) * feat_vec

    quad_scale = scale_vec[torch.arange(scale_vec.shape[0]),self.cuboid_edge.view(-1,1)].t()
    quad_scale = quad_scale.view(-1, *self.cuboid_edge.shape).contiguous()

    return quad_RT, quad_feat, quad_scale, cuboid_RT 

  def forward(self, x_3d, RT):
    """Render a set of 3D bounding boxes to 2D image
    
    Args:
        x_3d (dict): 3D primitives including poses parameters and features 
        RT (torch.FloatTensor): Camera poses 
    
    Returns:
        images (torch.FloatTensor): Rendered image, alpha and depth, NxCxHxW
        area (torch.FloatTensor): Mean area of projected cuboids, N
        center_depth (torch.FloatTensor): depth of the center coordinate of the bounding box, N
        cuboid_RTs (torch.FloatTensor): The poses of the cuboids in camera coordinate
    """
    N = x_3d['R'].shape[0]

    assert RT.shape[0] == N

    quad_RTs, quad_feats, quad_scales, cuboid_RTs = self.get_cuboids(x_3d)

    image, silh, depth, area, center_depth = self.project(RT,
            quad_RTs, quad_feats, quad_scales,
            sigma=self.sigma, gamma=self.gamma)
    images = torch.cat((image, silh, depth), dim=1) # (BxN_obj)xCxHxW

    cuboid_RTs = RT @ cuboid_RTs

    return images, area, center_depth, cuboid_RTs    # (BxN_obj)xCxHxW


class RendererPoint(RendererQuadMesh):
  """Point cloud renderer for "point"

  Render cuboids at the same time to observe rendered alpha and depth
  """
  def __init__(self, pcd, img_size=64, near_plane=0., far_plane=100., transform_feat=False):
    """Initialization
    
    Args:
        pcd (Primitive): PointCloud primitive class with basic configurations (n_points, transform_feat) 
    """
    pcd.texsize = 1
    super(RendererPoint, self).__init__(pcd, img_size=img_size, near_plane=near_plane, far_plane=far_plane)
  
    self.n_pts = pcd.n_points
    self.transform_feat = pcd.transform_feat

    self.project_point = dr.ProjectPoint2Image(self.K, self.im_width, self.im_height)

    self.pts_scale = torch.ones(self.n_pts,).float()

  def forward(self, x_3d, RT):
    """Render a set of 3D point clouds to 2D image
    
    Args:
        x_3d (dict): 3D primitives including poses parameters and features 
        RT (torch.FloatTensor): Camera poses 
    
    Returns:
        images (torch.FloatTensor): Rendered image, alpha and depth, NxCxHxW
        area (torch.FloatTensor): Mean area of projected cuboids, N
        center_depth (torch.FloatTensor): depth of the center coordinate of the bounding box, N
        cuboid_RTs (torch.FloatTensor): The poses of the cuboids in camera coordinate
    """

    N = x_3d['R'].shape[0]

    x_3d_cuboid = x_3d.copy()
    x_3d_cuboid['feature'] = x_3d_cuboid['feature'][:, :3].view(N, 1, 1, 3)
    image_cuboid, area, center_depth, cuboid_RTs = super().forward(x_3d_cuboid, RT)
    image_cuboid = image_cuboid.view(N, -1, *image_cuboid.shape[2:])

    if RT.shape[0] == N:
      pass
    elif (RT.dim()==3 and RT.shape[0]==1) or RT.dim()==2:
      RT = RT.view(1,3,4).repeat(N,1,1)
    else:
      raise ValueError('Invalid RT!')

    feature = x_3d['feature']
    pts_loc = (torch.sigmoid(feature[:, :3*self.n_pts]) - 0.5) * 2.0
    pts_feat = feature[:, 3*self.n_pts:].contiguous()

    pts_loc = pts_loc.view(N, self.n_pts, 3)
    pts_feat = pts_feat.view(N, self.n_pts, -1)

    # transform points from object-centric coordinate to world coordinate
    pts_3d = self.transform(x_3d, pts_loc)

    # transform feature vectors with cuboid parameters
    if self.transform_feat:
        r_mat = x_3d['R']
        if pts_feat.shape[2] % 3 != 0:
            raise ValueError('Point feature needs to be Cx3!')
        pts_feat = pts_feat.view(N, self.n_pts, -1, 3).flatten(1,2) # BxNxCx3 -> Bx(NxC)x3
        pts_feat = (r_mat @ pts_feat.transpose(1,2)).transpose(1,2) ## Only rotation, no translation 
        pts_feat = pts_feat.contiguous().view(N, self.n_pts, -1)
        
    # render point cloud
    image_pts = self.project_point(RT, pts_3d, 
            pts_feat, self.pts_scale.unsqueeze(0).expand(N, -1).to(RT.device))

    images = torch.cat((image_pts, image_cuboid[:,-2:]), dim=1) # (BxN_obj)xCxHxW

    return images, area, center_depth, cuboid_RTs # (BxN_obj)xCxHxW
