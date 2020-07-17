import argparse
import os
from os import path
import math
import pickle
import torch
from torch import nn
from gan_training.checkpoints import CheckpointIO
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (
  load_config
)
from controllable_gan.config import get_dataloader, get_out_dir, build_models
from controllable_gan.transforms import ObjectTranslation, ObjectRotation


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
  
  # initialize fid evaluators
  cache_file = os.path.join(out_dir, 'cache_test.npz')
  test_loader = get_dataloader(config, split='test')
  evaluator.inception_eval.initialize_target(test_loader, cache_file=cache_file)
  cache_file = os.path.join(out_dir, 'cache_test_single.npz')
  test_loader_single = get_dataloader(config, split='test', single=True)
  evaluator_single.inception_eval.initialize_target(test_loader_single, cache_file=cache_file)
  
  # Load checkpoint
  load_dict = checkpoint_io.load(model_file)
  it = load_dict.get('it', -1)
  epoch_idx = load_dict.get('epoch_idx', -1)

  # Pick a random but fixed seed
  seed = torch.randint(0, 10000, (1,))[0]


  # Evaluation Loop

  print('Computing FID score...')

  def sample(transforms, n_obj=None):
    while True:
      z = zdist.sample((evaluator.batch_size,))
      
      if n_obj is None:
        obj_masks = None
      else:
        n_fg = config['generator']['n_bbox']
        obj_masks = torch.zeros(evaluator.batch_size, n_fg, dtype=torch.bool)
        for obj_mask in obj_masks:
          obj_mask[torch.randperm(n_fg)[:n_obj]] = 1
        obj_masks = obj_masks.view(-1, 1, 1, 1)
      
      x = evaluator.create_samples(z, param_transforms=transforms, obj_masks=obj_masks)
      rgb = x['img'].view(z.shape[0], -1, *x['img'].shape[1:])[:, -1, 0]   # (BxN_tf)xN_objx3xHxw -> Bx3xHxW
      del x, z
      yield rgb

  transforms = {'none': [],
                'trans': [ObjectTranslation()],
                'rot': [ObjectRotation(axis='y', param_range=(0, 2*math.pi))],
                'single': []}

  fid = {}
  for name, tf in transforms.items():
    # reset the seed to ensure we sample identical z for all transforms
    torch.random.manual_seed(seed)
    
    if name == 'single':
      if generator.module.n_fg == 1:
        fid[name] = fid['none']
      else:
        fid[name] = evaluator.compute_fid_score(sample_generator=sample(tf, n_obj=1))
    
    else:
      fid[name] = evaluator.compute_fid_score(sample_generator=sample(tf))

  print('FID scores:\n\t{}'.format('\n\t'.join(f'{k}: {v:0.1f}' for k, v in fid.items())))

  out_file = os.path.join(out_dir, 'fid.pkl')
  with open(out_file, 'wb') as f:
    pickle.dump(fid, f)

  print('Saved FID to {}'.format(out_file))
