import os
import numpy as np
import yaml
import math
import torch
import torchvision
from torch.utils.data import DataLoader
from gan_training.config import toggle_grad

from .renderer import RendererQuadMesh, RendererPoint, \
                     RendererMesh
from .primitives import PointCloud, Cuboid, Mesh
from .transforms import ObjectRotation, ObjectTranslation
from .datasets import ObjectDataset
from .loss import SizeLoss, ConsistencyLoss

primitive_dict = {'point': # point cloud
                    {'type': PointCloud, 'renderer': RendererPoint, 'kwargs': {'n_points': 32, 'n_channel': 3}},
                  'cuboid_sr': # cuboid based on adapted Soft Rasterizer
                    {'type': Cuboid, 'renderer': RendererQuadMesh, 'kwargs': {'texsize': 7, 'n_channel': 8}},
                  'cuboid': # cuboid based on Neural Mesh Renderer 
                    {'type': Mesh, 'renderer': RendererMesh, 'kwargs': {'render_type': 'cuboid', 'n_channel': 8}},
                  'sphere': # sphere based on Neural Mesh Renderer
                    {'type': Mesh, 'renderer': RendererMesh, 'kwargs': {'render_type': 'sphere', 'n_channel': 8}},
                  }

def save_config(path, config):
  ''' Saves config.

  Args:
      path (str): path to output file
      config (dict): configurations
  '''
  with open(path, 'w') as f:
    yaml.safe_dump(config, f)


def build_primitives(config):
  """Return a primitive"""
  render_type = config['generator']['render_type']
  primitive = primitive_dict[render_type]['type']
  return primitive(**primitive_dict[render_type]['kwargs'])


def build_renderer(primitive, img_size, near_plane, far_plane, config):
  """Return a differentiable renderer"""
  renderer = primitive_dict[config['generator']['render_type']]['renderer']
  return renderer(primitive, img_size, near_plane=near_plane, far_plane=far_plane)


def build_optimizers(generator, discriminator, config):
  """Return optimizers of the generator and the discriminator"""
  lr_g = config['training']['lr_g']
  lr_d = config['training']['lr_d']
  
  toggle_grad(generator, True)
  toggle_grad(discriminator, True)
  
  g_params = generator.parameters()
  d_params = discriminator.parameters()
  
  # Optimizers
  g_optimizer = torch.optim.RMSprop(g_params, lr=lr_g, alpha=0.99, eps=1e-8)
  d_optimizer = torch.optim.RMSprop(d_params, lr=lr_d, alpha=0.99, eps=1e-8)
  
  return g_optimizer, d_optimizer


def build_g_losses(generator, config):
  """Return regularization losses on the generator"""
  losses_g = {}
  
  # compactness loss
  w_compact = config['training']['weight_compactness']
  l_compact = SizeLoss()
  if w_compact > 0:
    losses_g['compact'] = (w_compact, l_compact)
  
  losses_g_2d = {}
  
  # 3d consistency loss
  w_consistency = config['training']['weight_3dconsistency']
  kwargs = {
    'n_tf': generator.n_tf,  # do not count original here
    'w_depth': 1.,
    'w_rgb': 5.,
    'max_batchsize': 96,
    'K': generator.renderer.K,
    'imsize': generator.imsize,
    'clamp': 0.
  }
  l_consistency = ConsistencyLoss(**kwargs)
  if w_consistency > 0:
    losses_g_2d['3d_consist'] = (w_consistency, l_consistency)
  
  return losses_g, losses_g_2d


def build_models(config):
  """Return generator and discriminator"""
  from .model import Generator
  from externals.pytorch_spectral_normalization_gan.model_resnet import Discriminator
  
  param_transforms = build_param_transforms(config['training']['param_transforms'])
  # Build models
  generator = Generator(config, param_transforms)
  
  input_nc = 3
  nlabels = config['data']['nlabels']
  discriminator = Discriminator(input_nc, nlabels=nlabels)
  
  return generator, discriminator


def build_param_transforms(name):
  """Return transformation objects"""
  if name == 'none':
    return []
  tf_type, shift, n_tf = name.split('_')
  shift, n_tf = torch.tensor(float(shift), dtype=torch.float), int(n_tf)
  
  if tf_type == 'rot':
    tf = ObjectRotation(axis='y', param_range=(0, shift * math.pi / 180))
  elif tf_type == 'trans':
    tf = ObjectTranslation(axis='xz', param_range=(-shift, shift))
  else:
    raise AttributeError
  
  param_transforms = [tf for _ in range(n_tf)]
  return param_transforms


def get_out_dir(config):
  """Return output directory"""
  out_dir = config['training']['out_dir']
  
  # dataset info
  dataset_name = config['data']['name']
  
  # generator info
  nprim = config['generator']['n_prim']
  render_type = config['generator']['render_type']
  bg_cube = config['generator']['bg_cube']
  
  param_transforms = config['training']['param_transforms']
  weight_compactness = config['training']['weight_compactness']
  weight_3dconsistency = config['training']['weight_3dconsistency']
  
  out_dir = f'{out_dir}/{dataset_name}_bg{bg_cube:d}_{render_type}{nprim}_' + \
            f'sz{weight_compactness}_3d{weight_3dconsistency}_tf{param_transforms}_origin2'
  
  return out_dir


def get_dataloader(config, split='train', single=False):
  """Return data loader"""
  root_dir = config['data']['root_dir']
  sub_dirs = config['data']['sub_dirs']
  data_dirs = [os.path.join(root_dir, sub_dir) for sub_dir in sub_dirs]

  if not all([os.path.isdir(data_dir) for data_dir in data_dirs]):
      raise ValueError('Incorrect data path!')

  if single:
    data_dirs = [d for d in data_dirs if int((d.split('/')[-1]).split('_')[0][-1])==1]
  
  if split == 'train':
    nlabels = config['data']['nlabels']
  else:
    nlabels = 1
    
  imsize = config['data']['img_size']
  
  transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(imsize),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: x + 1. / 128 * torch.rand(x.size())),
    lambda x: x * 2 - 1
  ])
  
  loader_kwargs = dict(
    batch_size=config['training']['batch_size'],
    shuffle=True,
    pin_memory=False,
    sampler=None,
    drop_last=True
  )
  if nlabels > 1:
      loader_kwargs['batch_size'] *= 2      # compensate for pure background images

  if split == 'train':
    loader_kwargs['num_workers'] = config['training']['nworkers']
  else:
    loader_kwargs['num_workers'] = config['test']['nworkers']
  
  dataset = ObjectDataset(data_dirs, split, transforms=transforms, nlabels=nlabels)
  
  return DataLoader(dataset, **loader_kwargs)
