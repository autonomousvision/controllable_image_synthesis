import torch
from torch.nn import functional as F
from torch.nn import init
import numpy as np

class Camera(object):
  """Camera object for sampling camera extrinsics"""
  def __init__(self, RT=None, radius=None, range_u=(0.0, 1.0), range_v=(0.0, 1.0), device='cuda'):
    self.device = device
    
    self.radius = radius
    self.range_u = range_u
    self.range_v = range_v
    
    if RT is None:
      RT = torch.eye(3, 4, dtype=torch.float).to(self.device)
    if self.radius is not None:
      RT[2, 3] = self.radius
    self.RT = RT
  
  def _sample_extrinsics(self, n):
    """Sample n camera poses"""
    u = np.random.uniform(*self.range_u, n)
    v = np.random.uniform(*self.range_v, n)

    return get_cam_extrinsics(u, v, self.radius)
  
  def get_extrinsics(self, batch_size, random_sample=False):
    if random_sample:
      return self._sample_extrinsics(batch_size)
    
    if self.RT.ndimension() == 2 or (self.RT.ndimension()==3 and self.RT.shape[0]==1):
      return self.RT.repeat(batch_size, 1, 1)
    
    return self.RT

  def set_pose(self, azimuth, polar):
    """Set RT of camera to pose specified by azimuth and polar angle.
    Args:
        azimuth (float): Azimuth angle in rad.
        polar (float): Polar angle in rad.
    """
    # transform azimuth and polar angle to uv
    u = np.array([azimuth / (2 * np.pi)])
    v = np.array([polar / np.pi])

    self.RT = get_cam_extrinsics(u, v, self.radius)


def get_rotation_from_axis_angle(u, theta):
  # normalize axis
  u = u / u.norm(p=2)
  u_cross = torch.tensor([[0, -u[2], u[1]],
                          [u[2], 0, -u[0]],
                          [-u[1], u[0], 0]])    # cross product matrix
  u_op = u.ger(u)         # outer product
  cos_theta = torch.cos(theta)
  sin_theta = torch.sin(theta)
  return cos_theta * torch.eye(3).type_as(u) + sin_theta * u_cross + (1 - cos_theta) * u_op

def get_rotation_from_two_vecs(rotation):
  # rotation: Nx3x2
  rotvec1 = rotation[:,:,0] / torch.norm(rotation[:,:,0], dim=1, keepdim=True)
  rotvec2_proj = torch.sum(rotvec1 * rotation[:,:,1], dim=1, keepdim=True) * rotvec1 
  rotvec2 = rotation[:,:,1] - rotvec2_proj
  rotvec2 = rotvec2 / torch.norm(rotvec2, dim=1, keepdim=True)
  rotvec3 = torch.cross(rotvec1, rotvec2, dim=1)
  rotmat = torch.cat((rotvec1.view(-1,3,1), rotvec2.view(-1,3,1), rotvec3.view(-1,3,1)), dim=2)
  return rotmat

def look_at(eye, at=torch.tensor([0, 0, 0]), up=torch.tensor([0, 1, 0])):
  device = eye.device

  at = at.float().to(device).view(1,3)
  up = up.float().to(device).view(1,3)
  eye = eye.view(-1,3)
  up = up.repeat(eye.shape[0]//up.shape[0],1)

  z_axis = F.normalize(at - eye, eps=1e-5, dim=1)
  x_axis = F.normalize(torch.cross(up, z_axis), eps=1e-5, dim=1)
  y_axis = F.normalize(torch.cross(z_axis, x_axis), eps=1e-5, dim=1)
  
  r_mat = torch.cat((x_axis.view(-1,3,1), y_axis.view(-1,3,1), z_axis.view(-1,3,1)), dim=2)

  return r_mat


def get_cam_extrinsics(u, v, radius):
  theta = 2*np.pi*u
  phi = np.arccos(1-2*v) 
  cx = radius*np.sin(phi)*np.cos(theta) 
  cy = radius*np.sin(phi)*np.sin(theta) 
  cz = radius*np.cos(phi) 
  
  # y pointing to up in our coordinate 
  # https://www.ranjithraghunathan.com/blender-coordinate-system-to-opengl/
  tcam = torch.tensor([cx, -cz, cy]).float().cuda().transpose(0,1)  #Bx3
  rcam = look_at(tcam) # Bx3x3

  return torch.cat((rcam.transpose(1,2), -rcam.transpose(1,2) @ tcam.view(-1,3,1)), dim=2)

def get_cube_RT_single(scale=1.0, device='cuda'):
  if isinstance(scale, float):
      scale = torch.ones(3,).float().to(device)*scale
      #sx = sy = sz = scale
  elif (isinstance(scale, list) or isinstance(scale, torch.Tensor)) and len(scale)==3:
      sx = scale[0]; sy = scale[1]; sz = scale[2]
  else:
      raise NotImplementedError
  pts_3d = torch.tensor([[0, 0, -1],
                         [0, 0, 1],
                         [0, -1, 0],
                         [0, 1, 0],
                         [-1, 0, 0],
                         [1, 0, 0]], dtype=torch.float32).to(device) * scale.view(1,3)
  pts_norm = torch.tensor([[0, 0, -1],
                         [0, 0, 1],
                         [0, -1, 0],
                         [0, 1, 0],
                         [-1, 0, 0],
                         [1, 0, 0]], dtype=torch.float32)
  up_axis = torch.tensor([0.,0.,1.], dtype=torch.float32, requires_grad=False)
  y_axis = torch.tensor([0.,1.,0.], dtype=torch.float32, requires_grad=False)
  pts_rot = get_rotation_from_norm(pts_norm, up_axis, y_axis)
  quad_RT = torch.zeros(6,4,4).float()
  quad_RT[:,:3,:3] = pts_rot
  quad_RT[:,:3,3] = pts_3d
  quad_RT[:,3,3] = 1 
  return quad_RT.to(device)
  

def get_cube_RT(scales):
  quad_RT = get_cube_RT_single(device = scales.device)
  quad_RT = quad_RT.view(1,6,4,4).repeat(scales.shape[0],1,1,1)
  Rs = quad_RT[:,:,:3,:3]
  ts = quad_RT[:,:,:3,3] * scales.unsqueeze(1)
  constant = quad_RT[:,:,3:]
  quad_RT = torch.cat((torch.cat((Rs, ts.unsqueeze(3)), dim=3), constant), dim=2)
  return quad_RT

def get_rotation_from_norm_single(n, u, y):
  # n: unit vector, y-axis
  # referring to blender trackTo, set up axis as z 
  # y-axis
  n = n/torch.norm(n)

  # z-axis
  proj_ = u - (torch.dot(u,n)/torch.dot(n,n))*n
  # when normal vector is aligned with up axis, set proj to fixed direction
  if torch.norm(proj_).item()<1e-6:
    proj = y
  else:
    proj = proj_/torch.norm(proj_)

  # x-axis
  right = torch.cross(proj, n)
  right = right/torch.norm(right)

  rot_mat = torch.cat((right.view(-1,1), n.view(-1,1), proj.view(-1,1)), dim=1)

  return rot_mat.view(1,3,3)

def get_rotation_from_norm(normal, up_axis, z):
  rot_mat = []
  for idx in range(normal.shape[0]):
    rot_mat.append(get_rotation_from_norm_single(normal[idx], up_axis, z))
  rot_mat = torch.cat(rot_mat, dim=0)
  return rot_mat


def get_spherical_coords(X):
  # X is N x 3
  rad = np.linalg.norm(X, axis=1)
  # Inclination
  theta = np.arccos(X[:, 2] / rad)
  # Azimuth
  phi = np.arctan2(X[:, 1], X[:, 0])

  # Normalize both to be between [-1, 1]
  vv = (theta / np.pi) * 2 - 1
  uu = ((phi + np.pi) / (2*np.pi)) * 2 - 1
  # Return N x 2
  return np.stack([uu, vv],1)


def compute_uvsampler(verts, faces, tex_size=2):
  """
  For a given mesh, pre-computes the UV coordinates for
  F x T x T points.
  Returns F x T x T x 2
  """
  alpha = np.arange(tex_size, dtype=np.float) / (tex_size-1)
  beta = np.arange(tex_size, dtype=np.float) / (tex_size-1)
  import itertools
  # Barycentric coordinate values
  coords = np.stack([p for p in itertools.product(*[alpha, beta])])
  vs = verts[faces]
  # Compute alpha, beta (this is the same order as NMR)
  v2 = vs[:, 2]
  v0v2 = vs[:, 0] - vs[:, 2]
  v1v2 = vs[:, 1] - vs[:, 2]
  # F x 3 x T*2
  samples = np.dstack([v0v2, v1v2]).dot(coords.T) + v2.reshape(-1, 3, 1)
  # F x T*2 x 3 points on the sphere
  samples = np.transpose(samples, (0, 2, 1))

  # Now convert these to uv.
  uv = get_spherical_coords(samples.reshape(-1, 3))
  # uv = uv.reshape(-1, len(coords), 2)

  uv = uv.reshape(-1, tex_size, tex_size, 2)
  return uv

def init_weights(net, init_type='normal', init_gain=0.02):
  """
  Initialize network weights
  Args:
    net (torch.nn.Module): network to be initialized
    init_type (str): the name of an initialization method: normal | xavier | kaiming | orthogonal
    init_gain (float): scaling factor for normal, xavier and orthogonal

  """
  def init_func(m):  # define the initialization function
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
      if init_type == 'normal':
        init.normal_(m.weight.data, 0.0, init_gain)
      elif init_type == 'xavier':
        init.xavier_normal_(m.weight.data, gain=init_gain)
      elif init_type == 'kaiming':
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
      elif init_type == 'orthogonal':
        init.orthogonal_(m.weight.data, gain=init_gain)
      else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
      if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
      init.normal_(m.weight.data, 1.0, init_gain)
      init.constant_(m.bias.data, 0.0)

  print('initialize network with %s' % init_type)
  net.apply(init_func)  # apply the initialization function <init_func>

