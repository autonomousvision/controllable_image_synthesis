import numpy as np
import torch
import collections
from utils.depth_map_visualization import color_depth_map
from matplotlib import cm


def color_depth_maps(depth, depth_max=None, rescale=True):
  
  depth = depth.detach().cpu().numpy()
  depth = depth / 2.0 + 0.5
  depth_max = 1.0
  #if depth_max is None:
  #  depth_max = 0.05
  #  # inifinity = depth.max()
  #  # depth_max = depth[depth<inifinity].max()
  if rescale:
    depth = (depth - depth.min())/(depth.max()-depth.min())
  
  colors = [color_depth_map(depth[i], depth_max)[None] for i in range(depth.shape[0])]
  colors = np.concatenate(colors, axis=0)
  colors = torch.from_numpy(colors).float().cuda()
  colors = colors.permute(0,3,1,2) # NxHxWxC -> NxCxHxW
  colors = colors.float() / 255.0
  
  return colors, depth_max
  

def draw_box(img, mask, width=2, c=torch.tensor([0.75, 0., 0.])):
  """
  Draw bounding box according to mask into image clone.
  Args:
      img (torch.FloatTensor or torch.ByteTensor): image in which to draw, CxHxW
      mask (torch.BoolTensor): object mask, 1xHxW
      width (int): width of the drawn box
      c (torch.FloatTensor): RGB color of the box

  Returns:

  """
  if img.dtype == torch.uint8:
    c = c * 255
  c = c.type_as(img)
  
  img = img.clone()  # do not modify original
  
  idcs = torch.nonzero(mask.squeeze(0))
  if len(idcs) == 0:
    return img
  h0, w0 = idcs.min(dim=0)[0]
  h1, w1 = idcs.max(dim=0)[0]
  
  # ad buffer for frame
  h0 = max(0, h0 - width)
  w0 = max(0, w0 - width)
  h1 = min(img.shape[-2] - 1, h1 + width)
  w1 = min(img.shape[-1] - 1, w1 + width)
  
  # impaint image
  img[..., h0:h0 + width, w0:w1] = c.view(3, 1, 1).repeat(1, width, w1 - w0)
  img[..., h1 - width + 1:h1 + 1, w0:w1] = c.view(3, 1, 1).repeat(1, width, w1 - w0)
  img[..., h0:h1, w0:w0 + width] = c.view(3, 1, 1).repeat(1, h1 - h0, width)
  img[..., h0:h1, w1 - width + 1:w1 + 1] = c.view(3, 1, 1).repeat(1, h1 - h0, width)
  
  return img

def draw_box_batch(imgs, masks, width=2, c=torch.tensor([0.75, 0., 0.])):
  return torch.stack([draw_box(img, mask, width, c) for img, mask in zip(imgs, masks)])

def wrap_images(imgs, colors):
  """Wrap images with color frame"""
  assert(imgs.shape[1]==3 and len(colors)==3)
  imgs[:,:,:2,:] = colors.view(1,3,1,1)
  imgs[:,:,-2:,:] = colors.view(1,3,1,1)
  imgs[:,:,:,:2] = colors.view(1,3,1,1)
  imgs[:,:,:,-2:] = colors.view(1,3,1,1)
  return imgs

def imdict_to_img(img_dict, n_per_row=8, n_rows=1):
  """Returns an image where each row corresponds to a key in the dictionary.
  Values are expected to be of format BxCxHxW.
  """
  od = collections.OrderedDict(sorted(img_dict.items()))
  imgs = torch.stack(list(od.values()), dim=0)      # AxBxCxHxW
  # make the rows
  imgs = torch.stack([imgs[:, i*n_per_row:(i+1)*n_per_row] for i in range(n_rows)], dim=0)    # n_rows x A x n_per_row x CxHxW
  imgs = imgs.flatten(0,2)
  return imgs

def get_cmap(n_fg):
  """Generate a color map for visualizing foreground objects
  
  Args:
      n_fg (int): Number of foreground objects 
  
  Returns:
      cmaps (numpy.ndarray): Colormap 
  """
  cmap = cm.get_cmap('Set1')
  cmaps = []
  for i in range(n_fg):
      cmaps.append(np.asarray(cmap(i))[:3])
  cmaps = np.vstack(cmaps)
  return cmaps


def visualize_objects(x, generator):
  """Visualize primitives and colored alpha maps of objects."""
  bg_color = 0.
  bs = x['img'].shape[0] // generator.n_tf
  obj_masks = x['obj_masks']
  out = {}

  def compose(rgb, alpha, depth):
    img_fg = rgb * alpha
    if generator.primitive_type != 'point':
      img_bg = bg_color * torch.ones(bs * generator.n_tf, 3, *generator.imsize).type_as(img_fg)
      img_fuse, _ = generator.alpha_composition(img_fg, img_bg, alpha, depth)
    else:
      img_fuse = torch.sum(img_fg.view(bs * generator.n_tf, generator.n_fg, 3, *generator.imsize), dim=1)
    return img_fuse

  # fused primitives
  rgb = x['prim'][:, :3] / 2 + 0.5
  alpha = (x['prim'][:, -2:-1] / 2 + 0.5) * obj_masks
  depth = x['prim'][:, -1:] / 2 + 0.5
  
  # (BxNxN_tf)x... -> (BxN_tfxN)x...
  reshape = lambda x: x.view(bs, generator.n_fg, generator.n_tf, *x.shape[1:]).transpose(1, 2).flatten(0, 2)
  rgb = reshape(rgb)
  alpha = reshape(alpha)
  depth = reshape(depth)
  
  out['vis_prim'] = compose(rgb, alpha, depth) * 2 - 1
  
  # colored fused alphas
  cmap = torch.from_numpy(get_cmap(generator.n_fg)).float().to(rgb.device)
  rgb = cmap.repeat(bs*generator.n_tf, 1).view(bs*generator.n_tf*generator.n_fg, 3, 1, 1)
  alpha = (x['layers_alpha'] / 2 + 0.5) * obj_masks

  # (BxNxN_tf)x... -> (BxN_tfxN)x...
  alpha = reshape(alpha)
  
  out['vis_layers_alpha'] = compose(rgb, alpha, depth) * 2 - 1

  # visualize primitives multiplied by alpha
  alpha = x['layers_alpha'] / 2 + 0.5
  rgb = x['layers_rgb'] / 2 + 0.5
  rgb = rgb * alpha

  # wrap each primitive with colored frame
  cmap = torch.from_numpy(get_cmap(generator.n_fg)).float().to(rgb.device)
  # (BxNxN_tf)x... -> (BxN_tf)xNx...
  rgb = reshape(rgb).view(-1, generator.n_fg, *rgb.shape[1:])
  rgb = torch.cat([wrap_images(rgb[:, i], cmap[i]).unsqueeze(1) for i in range(rgb.shape[1])], 1)
  # (BxN_tf)xN... -> (BxNxN_tf)x...
  reshape = lambda x: x.view(bs, generator.n_tf, generator.n_fg, *x.shape[2:]).transpose(1, 2).flatten(0, 2)
  rgb = reshape(rgb)
  x['layers_rgb'] = rgb * 2 - 1
  
  return out

def visualize_scenes(x, generator):
  """Split images into rgb and depth."""
  keys = ['img', 'img_single']
  if generator.bg_cube:
    keys.append('img_bg')
  
  out = {}
  for k in keys:
    v = x[k]
    if v is None:
      rgb = d = None
    else:
      rgb = x[k][:, :3]
      d = x[k][:, -1:]
      
    out[f'vis_{k}_rgb'] = rgb
    out[f'vis_{k}_depth'] = d
  
  return out

def colorize_primitives(x, generator):
  """rerender primitives with instance color for visualization"""
  bs = x['img'].shape[0] // generator.n_tf
  cmap = torch.from_numpy(get_cmap(generator.n_fg)).float().to(x['img'].device)
  
  if isinstance(generator.renderer, RendererMesh):
    renderer_orig = deepcopy(generator.renderer)
    generator.renderer.pred_uv = False
    generator.renderer.renderer.light_intensity_ambient = 0.6
    generator.renderer.renderer.light_intensity_directional = 0.5
    generator.renderer.renderer.light_color_ambient = torch.ones(generator.renderer.texture_channel, ).cuda()
    generator.renderer.renderer.light_color_directional = torch.ones(generator.renderer.texture_channel, ).cuda()
  
  # replace first three channels with color
  x = x.copy()    # do not modify original

  def reshape(feature):
    if generator.primitive_type == 'point':
      return feature.view(bs, generator.n_fg, generator.renderer.n_pts, -1)
    if generator.primitive_type == 'cuboid_sr':
      return feature.view(bs, generator.n_fg, generator.renderer.texsize ** 2*6, -1)
    if generator.primitive_type == 'cuboid' or generator.primitive_type == 'sphere':
      return feature
    raise ValueError('Unknown render type!')
  
  feature = reshape(x['feature'])   # BxN_objx...xC
  ch_start = 0
  if generator.primitive_type == 'point':
    ch_start = 3
  for idx in range(generator.n_fg):
    feature[:, idx, ..., ch_start:ch_start+3] = cmap[idx]
  
  x['feature'] = feature.view_as(x['feature'])
  x = generator.render_primitives(x)
  
  if isinstance(generator.renderer, RendererMesh):   # reset renderer if necessary
    generator.renderer = renderer_orig
  return x['prim']
