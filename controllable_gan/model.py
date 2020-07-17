import torch
from copy import deepcopy
from torch import nn
from .models import Generator2D, Generator3D
from .config import build_renderer, build_primitives
from .renderer import RendererBG, RendererMesh
from utils.commons import Camera, init_weights

class Generator(nn.Module):
  """Generator composed of 3D Genertor, Differentiable Renderer and 2D Generator """
  def __init__(self, config, param_transforms=[]):
    """Initialization of the generator
    
    Args:
        config (dict): Dictionary of configurations 
        param_transforms (list, optional): List of random transformations applied to 3D primitives. Defaults to None.
    """
    super(Generator, self).__init__()

    self.param_transforms = param_transforms
    
    self.n_fg = self.n_prim = config['generator']['n_prim']
    self.bg_cube = config['generator']['bg_cube']
    if self.bg_cube:
        self.n_prim += 1
    else:
        self.bg_color = 1.

    # primitive_type and n_tf are used only for visualization
    self.primitive_type = config['generator']['render_type']
    self.n_tf = len(param_transforms) + 1       # include original

    # randomly sample camera poses
    self.cam_transforms = True

    z_dim = config['z_dist']['dim']

    # initialize 3d generator
    n_hidden_bg = 128
    primitive = build_primitives(config)
    self.generator_3d = Generator3D(z_dim, self.n_fg, primitive, self.bg_cube, n_hidden_bg=n_hidden_bg)

    # initialize camera for rendering
    cam_radius = 2.0  # ensure camera radius is compatible with maximal translation (here: 1.0) of primitives
    self.camera = Camera(radius=cam_radius, range_u=(0.0, 1.0), range_v=(0.15, 0.371))  # same settings as in Blender

    # initialize differentiable renderer
    near_plane = 0.1
    far_plane = 4.0  # ensure background depth is larger than maximal translation of primitives (here: 1.0) and camera radius (here: 2.0)
    img_size = config['data']['img_size']
    self.renderer = build_renderer(primitive, img_size, near_plane, far_plane, config)
    self.imsize = (self.renderer.im_height, self.renderer.im_width)

    # initialize 2d generator
    d_generator2d_in = primitive.n_channel  # primitive channels
    d_generator2d_out = 5  # rgb + alpha + depth
    self.d_generator2d_in = d_generator2d_in
    self.generator_2d = Generator2D(d_generator2d_in, d_generator2d_out, n_prim=self.n_fg, zdim=z_dim)
    init_weights(self.generator_2d,  init_type='xavier', init_gain=0.02)

    # background generation network
    if self.bg_cube:
      self.bg_renderer = RendererBG(bg_radius=cam_radius, texture_channel=primitive.n_channel, near_plane=near_plane, far_plane=far_plane) # set bg radius equal to camera radius
      self.generator_2d_bg = Generator2D(d_generator2d_in, d_generator2d_out, n_prim=1, zdim=z_dim)
      init_weights(self.generator_2d_bg,  init_type='xavier', init_gain=0.02)

  def forward(self, z, y=None, obj_masks=None):
    """Generate composite images from a noise vector"""
    x_3d = self.forward_3d(z)
    x_2d = self.forward_2d(x_3d, z, obj_masks=obj_masks)
    x = dict(**x_3d, **x_2d)

    if len(self.param_transforms) > 0:
      # save original camera settings before modifying
      cam_transforms_orig = self.cam_transforms
      RT_cam_orig = self.camera.RT
      
      self.cam_transforms = False
      self.camera.RT = x['cam_RT']  # ensure we have the same RT
      obj_masks = x['obj_masks']

      for tf in self.param_transforms:
        x_3d_tf = x_3d.copy()
        tf(x_3d_tf)   # transform is in-place
        x_2d = self.forward_2d(x_3d_tf, z, obj_masks)
        x_tf = dict(**x_3d, **x_2d)
        self.append_output(x, x_tf)

      # restore original settings
      self.cam_transforms = cam_transforms_orig
      self.camera.RT = RT_cam_orig
    
    return x

  def forward_3d(self, z):
    """Generate 3D primitives from a noise vector"""
    return self.generator_3d(z)
    
  def forward_2d(self, x, z=None, obj_masks=None):
    """Generate composite images given 3D primitives"""
    x_prim = self.render_primitives(x)
    x_layers = self.generate_2d_images(x_prim, z)
    x_fused = self.compose_images(dict(**x_prim, **x_layers), obj_masks)
    
    out = dict(**x_prim, **x_layers, **x_fused)
    return out

  def render_primitives(self, x):
    """Project 3D primitives to the 2D image domain"""
    bs = x['R'].shape[0] // self.n_fg
  
    cam_RT = self.camera.get_extrinsics(bs, random_sample=self.cam_transforms).to(x['R'].device)
    cam_RTs = cam_RT.unsqueeze(1).repeat(1,self.n_fg, 1,1).flatten(0,1)
    prim, area, center_depth, prim_RT = self.renderer(x, cam_RTs)
  
    # rendered depths are in camera system of renderer
    # we now go to relative depth coordinates in [0, 1] by dividing with the background depth
    center_depth = center_depth / self.renderer.far_plane
    prim[:, -1] = prim[:, -1] / self.renderer.far_plane
  
    # normalize center alpha and depth values
    prim[:, -2:] = prim[:, -2:] * 2 - 1 

    if self.bg_cube:  # render background primitive separately
      x_bg = x['bg']
      prim_bg = self.bg_renderer(x_bg, cam_RT)
      prim_bg[:, -1] = prim_bg[:, -1] / self.renderer.far_plane
      prim_bg[:, -2:] = prim_bg[:, -2:] * 2 - 1
    else:
      prim_bg = None

    out = {
      'cam_RT': cam_RT,
      'prim': prim,
      'prim_bg': prim_bg,
      'prim_RT': prim_RT,
      'prim_center_depth': center_depth,
      'prim_area': area,
    }
    return out

  def generate_2d_images(self, x, z=None):
    """Convert each rendered primitive into a 2D image individually"""
    prim = x['prim']
    center_depth = x['prim_center_depth']

    # transform the global depth to local depth
    depth = prim[:, -1:]
    depth = self.depth_local_to_global(depth, center_depth, inverse=True)
    prim = torch.cat([prim[:, :-1], depth], dim=1)

    img = self.generator_2d(prim[:,:self.d_generator2d_in], z)
    
    # split into RGB, alpha and depth
    rgb = img[:, :3]
    alpha = img[:, 3:4]
    depth = img[:, 4:5]

    # combine predicted alpha with rendered alpha of primitive
    alpha_prim = prim[:, -2:-1]
    # for multiplication go from [-1, 1] to [0,1]
    alpha = alpha / 2 + 0.5
    alpha_prim = alpha_prim / 2 + 0.5

    alpha = alpha * alpha_prim
    mask = alpha > 0.5
    alpha = alpha * 2 - 1  # go back to [-1, 1]

    # convert local depth prediction back to global depth
    depth = self.depth_local_to_global(depth, center_depth)
    
    img_bg = None
    if self.bg_cube:    # generate background separately
      prim_bg = x['prim_bg']
      img_bg = self.generator_2d_bg(prim_bg[:,:self.d_generator2d_in], z)   # BxCxHxW
      
      bg_depth = torch.ones_like(img_bg[:, 4:5])

      img_bg = torch.cat([img_bg[:, :4], bg_depth], dim=1)
   
    out = {
      'layers_rgb': rgb,
      'layers_alpha': alpha,
      'layers_mask': mask,
      'layers_depth': depth,
      'layers_depth_rescaled': depth * self.renderer.far_plane,          # save this for consistency loss
      'img_bg': img_bg,
    }
    return out

  def depth_local_to_global(self, depth, center_depth, inverse=False):
    """Interpret depth as offset from the center depth of the primitive.
    Convert local offset back to global depth."""
  
    # choose maximum offfset as the (normalized) maximum size of the bounding box
    # (this prevents getting to the saturated region of the tanh)
    max_offset = 2 * self.generator_3d.prim_size_max / self.renderer.far_plane
  
    if not inverse:
      # rescale from [-1, 1] -> [-max_offset, max_offset]
      depth = depth * max_offset
    
      # shift according to the center depth of the primitive
      depth = depth + center_depth.view(*center_depth.shape, 1, 1, 1)
    
    else:
      # shift according to the center depth of the primitive
      depth = depth - center_depth.view(*center_depth.shape, 1, 1, 1)
      
      # rescale from [-max_offset, max_offset] -> [-1, 1]
      depth = depth / max_offset
  
    return depth

  def compose_images(self, x, obj_masks=None):
    """Compose the output of the 2D generator to obtain to the composite image"""
    bs = x['layers_rgb'].shape[0] // self.n_fg

    # [-1,1] -> [0,1]
    rgb = x['layers_rgb'] / 2 + 0.5
    alpha = x['layers_alpha'] / 2 + 0.5
    depth = x['layers_depth'] / 2 + 0.5
    mask = x['layers_mask'].float()
  
    # (soft) masking of RGB images with alpha values
    rgb = rgb * alpha
    # (hard) masking of depth with alpha masks, s.t. depth value is not affected by alpha multiplication
    depth = depth * mask
    
    # mask out objects
    if obj_masks is not None:
      pass
    else:       # randomly mask out foreground objects
      obj_masks = self.get_random_mask(bs).view(bs*self.n_fg, 1, 1, 1)
    obj_masks = obj_masks.type_as(alpha)
    
    alpha_masked = alpha * obj_masks
    rgb_masked = rgb * obj_masks
    
    # get background
    if self.bg_cube:
      rgb_bg = x['img_bg'][:, :3] / 2 + 0.5
    else:         # use constant value
      rgb_bg = self.bg_color * torch.ones(bs, 3, *self.imsize).type_as(rgb)

    # fuse RGB
    prim_depth = x['prim'][:, -1:] / 2 + 0.5      # use primitive depth for ordering objects
    rgb_fuse, rgb_fg = self.alpha_composition(rgb_masked, rgb_bg, alpha_masked, prim_depth)
    # for single object discriminator: render each object separately with background
    rgb_single = rgb + (1 - alpha) * rgb_bg.repeat(self.n_fg, 1, 1, 1)
   
    # fuse depth and append to rgb
    if self.bg_cube:
      depth_bg = x['img_bg'][:, -1:] / 2 + 0.5
    else:
      depth_bg = torch.ones(bs, 1, *self.imsize).type_as(depth)
    
    depth_fuse = self.fuse_depth(depth, depth_bg, mask)
    depth_single = depth + (1-mask) * depth_bg.repeat(self.n_fg, 1, 1, 1)

    img = torch.cat([rgb_fuse, depth_fuse], dim=1)
    img_single = torch.cat([rgb_single, depth_single], dim=1)
    
    # rescale from [0, 1] back to [-1, 1]
    img = img * 2 - 1
    img_single = img_single * 2 - 1
    
    out = {
      'img': img,
      'img_single': img_single,
      'obj_masks': obj_masks
    }
    return out

  def get_random_mask(self, bs):
    """Generate a binary mask to randomly mask out foreground objects"""
    obj_mask = torch.ones(bs, self.n_fg, dtype=torch.bool)
    n_masked = torch.randint(0,self.n_fg, (bs,))
    for obj_mask_i, n in zip(obj_mask, n_masked):
      idx_masked = torch.randperm(self.n_fg)[:n]
      obj_mask_i[idx_masked] = 0
    
    return obj_mask

  def fuse_depth(self, depth, depth_bg, alpha):
    """Select pixels with minimal depth."""
    depth = depth + (1-alpha) * depth_bg.repeat(self.n_fg, 1, 1, 1)
    depth = depth.view(-1, self.n_fg, 1, *self.imsize)      # (BxN_obj)x1xHxW -> BxN_objx1xHxW
    depth = torch.sort(depth, dim=1)[0][:, 0]
    return depth

  def alpha_composition(self, img_fg, img_bg, alpha_fg, depth_fg):
    """Alpha composition"""
    bs = int(img_fg.shape[0] / self.n_fg)
    h, w = self.imsize
    
    # (BxN_obj)xCxHxW -> Cx(BxHxW)xN_obj
    reshape = lambda x: x.view(bs, self.n_fg, -1, h*w).permute(2, 0, 3, 1).flatten(1,2)
    img_fg = reshape(img_fg)
    alpha_fg = reshape(alpha_fg)
    depth_fg = reshape(depth_fg)

    depth_order = torch.argsort(depth_fg.squeeze(0), dim=1)
    
    # initialize with zeros
    img_fuse = torch.zeros_like(img_fg[:, :, 0])
    alpha_fuse = torch.zeros_like(alpha_fg[:, :, 0])

    pixel_idx = torch.arange(bs * h * w).long().to(img_fg.device)
    # sequentially add values
    for i in range(self.n_fg):
      obj_idx = depth_order[:, i]
      img_fuse = img_fuse + img_fg[:, pixel_idx, obj_idx] * (1 - alpha_fuse)
      alpha_fuse = alpha_fuse + alpha_fg[:, pixel_idx, obj_idx] * (1 - alpha_fuse)

    img_fuse = img_fuse.view(-1, bs, h, w).permute(1, 0, 2, 3).contiguous()                 # Cx(BxHxW) -> BxCxHxW
    alpha_fuse = alpha_fuse.view(-1, bs, h, w).permute(1, 0, 2, 3).contiguous()             # 1x(BxHxW) -> Bx1xHxW

    img_fg = img_fuse

    # alpha composition for background
    alpha_bg = 1 - alpha_fuse
    img_fuse = img_fuse + alpha_bg * img_bg

    return img_fuse, img_fg
  
  @staticmethod
  def append_output(x, x_tf):
    """Append transformed output next to the original."""
    assert x.keys() == x_tf.keys()
    for k, v_tf in x_tf.items():
      v = x[k]
      if v is None: continue
      
      if v.shape[0] != v_tf.shape[0]:     # append multiple times to one key
        v = v.view(v_tf.shape[0], -1, *v_tf.shape[1:])      # (BxN)xN_tfx...
        v_tf = v_tf.unsqueeze(1)
        v = torch.cat([v, v_tf], dim=1)
      else:
        v = torch.stack([v, v_tf], dim=1)
      
      x[k] = v.flatten(0, 1)
    return x
  
