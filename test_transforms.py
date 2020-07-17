import argparse
import os
from os import path
import math
import torch
from torch import nn
from functools import partial
from gan_training.checkpoints import CheckpointIO
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (
  load_config
)
from controllable_gan.config import get_dataloader, get_out_dir, build_models
from controllable_gan.transforms import ObjectTranslation, ObjectRotation
from utils.io_utils import save_tensor_images


def save_imdict(im_dict, out_dir):
  for vis_attr_i in ('vis_img_rgb', 'vis_layers_alpha', 'vis_prim'):
    imgs = im_dict[vis_attr_i]
    imgs = imgs.view(sample_size, -1, *imgs.shape[2:])  # (BxN_tf)x1x... -> BxN_tfx...
    imgs = imgs[:, 1:]    # first sample is without transform
    imgs = imgs / 2 + 0.5  # [-1,1] -> [0,1]

    prefix = os.path.join(out_dir, vis_attr_i)
    save_tensor_images(imgs, prefix, nrow=sample_nrow, as_row_image=True, as_gif=True)


if __name__ == '__main__':
  # Arguments
  parser = argparse.ArgumentParser(
    description='Compute FID for a trained 3D controllable GAN.'
  )
  parser.add_argument('config', type=str, help='Path to config file.')

  args = parser.parse_args()
  
  config = load_config(args.config, 'configs/default.yaml')
  is_cuda = (torch.cuda.is_available())
  assert is_cuda, 'No GPU device detected!'

  # Shorthands
  nlabels = config['data']['nlabels']
  batch_size = config['test']['batch_size']

  out_dir = get_out_dir(config)
  checkpoint_dir = path.join(os.path.dirname(args.config), 'chkpts')
  out_dir = path.join(out_dir, 'test')
  
  # Creat missing directories
  os.makedirs(out_dir, exist_ok=True)
  
  # Logger
  checkpoint_io = CheckpointIO(
    checkpoint_dir=checkpoint_dir
  )
  
  device = torch.device("cuda:0" if is_cuda else "cpu")

  # Disable parameter transforms for testing
  config['training']['param_transforms'] = 'none'
  
  # Create models
  generator, _ = build_models(config)
  print(generator)

  # Disable random sampling of primitives


  # Put models on gpu if needed
  generator = generator.to(device)

  # Use multiple GPUs if possible
  generator = nn.DataParallel(generator)

  # Register modules to checkpoint
  checkpoint_io.register_modules(
    generator=generator,
  )

  # Get model file
  model_file = config['test']['model_file']

  # Distributions
  ydist = get_ydist(nlabels, device=device)
  zdist = get_zdist('gauss', config['z_dist']['dim'],
                    device=device)

  # Test generator
  generator_test = generator

  # Evaluator
  evaluator = Evaluator(generator_test, zdist, batch_size=config['test']['batch_size'], device=device)
  evaluator_single = Evaluator(generator_test, zdist, batch_size=config['test']['batch_size'], device=device)
  
  # Load checkpoint
  load_dict = checkpoint_io.load(model_file)
  it = load_dict.get('it', -1)
  epoch_idx = load_dict.get('epoch_idx', -1)

  # Pick a random but fixed seed
  seed = torch.randint(0, 10000, (1,))[0]


  # Evaluation Loop

  sample_size = config['test']['sample_size']
  sample_nrow = n_steps = config['test']['sample_nrow']
  ztest = zdist.sample((sample_size,))

  # Always mask out the same primitives
  obj_masks = generator_test.module.get_random_mask(sample_size).view(sample_size*generator_test.module.n_fg, 1, 1, 1)

  # Create mask to modify only one primitive at a time
  idx_modified = []
  unmasked_idcs = torch.where(obj_masks.view(sample_size, generator_test.module.n_fg, 1, 1, 1))
  for i in range(sample_size):      # select index only from not masked primitives
    choices = unmasked_idcs[1][unmasked_idcs[0] == i]
    idx_modified.append(choices[torch.randperm(len(choices))[:1]])
  idx_modified = torch.cat(idx_modified)
  mask_modified = torch.zeros((sample_size, config['generator']['n_prim']), dtype=torch.bool)
  mask_modified[range(sample_size), idx_modified] = 1

  # Translations
  print('Sample object translations...')

  t_range = (-0.8, 0.8)

  for axis in ('x', 'z'):
    odir = os.path.join(out_dir, 'translation', axis)
    os.makedirs(odir, exist_ok=True)

    transform = ObjectTranslation(axis=axis)
    transforms = [partial(transform, value=t, axis=axis, mask=mask_modified)
                  for t in torch.linspace(*t_range, n_steps)]

    im_dict = evaluator.create_samples(ztest, transforms, obj_masks=obj_masks)
    save_imdict(im_dict, odir)

  # Rotations
  print('Sample object rotations...')

  r_range = torch.tensor([0, 2*math.pi])

  odir = os.path.join(out_dir, 'rotation')
  os.makedirs(odir, exist_ok=True)

  transform = ObjectRotation(axis='y')
  transforms = [partial(transform, value=r, mask=mask_modified)
                for r in torch.linspace(*r_range, n_steps+1)[:-1]]    # discard last step due to periodicity

  im_dict = evaluator.create_samples(ztest, transforms, obj_masks=obj_masks)
  save_imdict(im_dict, odir)

  # Azimuth
  print('Sample camera rotations...')

  a_range = (0., 2.*math.pi)
  polar = 0.25*math.pi

  odir = os.path.join(out_dir, 'azimuth')
  os.makedirs(odir, exist_ok=True)

  transforms = [partial(lambda x, a, p: generator_test.module.camera.set_pose(azimuth=a, polar=p), a=a, p=p)
                for a, p in zip(torch.linspace(*a_range, n_steps+1)[:-1], torch.ones(n_steps) * polar)]     # discard last step due to periodicity

  im_dict = evaluator.create_samples(ztest, transforms, obj_masks=obj_masks)
  save_imdict(im_dict, odir)

  # Polar
  print('Sample camera elevations...')

  azimuth = 0
  p_range = (0.15*math.pi, 0.371*math.pi)

  odir = os.path.join(out_dir, 'polar')
  os.makedirs(odir, exist_ok=True)

  transforms = [partial(lambda x, a, p: generator_test.module.camera.set_pose(azimuth=a, polar=p), a=a, p=p)
                for a, p in zip(torch.ones(n_steps) * azimuth, torch.linspace(*p_range, n_steps))]
  
  im_dict = evaluator.create_samples(ztest, transforms, obj_masks=obj_masks)

  im_dict = evaluator.create_samples(ztest, transforms, obj_masks=obj_masks)
  save_imdict(im_dict, odir)
