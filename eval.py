import argparse
import os
from os import path
import pickle
import torch
from torch import nn
import math
from gan_training.checkpoints import CheckpointIO
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (
  load_config, build_models
)
from controllable_gan.config import get_dataloader, get_out_dir
from controllable_gan.transforms import ObjectTranslation, ObjectRotation
from utils.io_utils import save_tensor_images
from utils.visualization import draw_box_batch


if __name__ == '__main__':
  # Arguments
  parser = argparse.ArgumentParser(
    description='Test a trained 3D controllable GAN and create visualizations.'
  )
  parser.add_argument('config', type=str, help='Path to config file.')
  parser.add_argument('--eval-attr', type=str, default='fid,rot,trans,cam', help='Attributes to evaluate.')

  args = parser.parse_args()
  
  config = load_config(args.config, 'configs/default.yaml')
  eval_attr = args.eval_attr.split(',')
  is_cuda = torch.cuda.is_available()
  assert is_cuda, 'No GPU device detected!'
  
  # Shorthands
  nlabels = config['data']['nlabels']
  batch_size = config['test']['batch_size']
  sample_size = config['test']['sample_size']
  sample_nrow = config['test']['sample_nrow']
  
  out_dir = get_out_dir(config)
  checkpoint_dir = path.join(out_dir, 'chkpts')
  out_dir = path.join(out_dir, 'test')
  
  # Creat missing directories
  os.makedirs(out_dir, exist_ok=True)
  
  # Logger
  checkpoint_io = CheckpointIO(
    checkpoint_dir=checkpoint_dir
  )
  
  device = torch.device("cuda:0" if is_cuda else "cpu")

  # Disable parameter transforms for testing
  config['param_transforms'] = 'none'
  
  # Create models
  generator, discriminator = build_models(config)
  print(generator)
  print(discriminator)
  
  # Put models on gpu if needed
  generator = generator.to(device)
  discriminator = discriminator.to(device)
  
  # Use multiple GPUs if possible
  generator = nn.DataParallel(generator)
  discriminator = nn.DataParallel(discriminator)
  
  # Register modules to checkpoint
  checkpoint_io.register_modules(
    generator=generator,
    discriminator=discriminator,
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
  if 'fid' in eval_attr:
    cache_file = os.path.join(out_dir, 'cache_test.npz')
    test_loader = get_dataloader(config, split='test')
    evaluator.inception_eval.initialize_target(test_loader, cache_file=cache_file)
    cache_file = os.path.join(out_dir, 'cache_test_single.npz')
    test_loader_single = get_dataloader(config, split='test', single=True)
    evaluator_single.inception_eval.initialize_target(test_loader_single, cache_file=cache_file)
  
  # Load checkpoint if existant
  load_dict = checkpoint_io.load(model_file)
  it = load_dict.get('it', -1)
  epoch_idx = load_dict.get('epoch_idx', -1)

  # Pick a random but fixed seed
  seed = torch.randint(0, 10000, (1,))[0]
  
  if 'fid' in eval_attr:
    print('Computing FID score...')

    def sample(transform):
      while True:
        z = zdist.sample((evaluator.batch_size,))
        x = evaluator.create_samples(z, param_transform=transform)
        rgb = x['img']
        del x, z
        yield rgb
    
    transforms = {'none': None,
                  'trans': ObjectTranslation(),
                  'rot': ObjectRotation(axis='y', type='extrinsic')}
    
    fid = {}
    for name, tf in transforms.items():
      # reset the seed to ensure we sample identical z for all transforms
      torch.random.manual_seed(seed)
      fid[name] = evaluator.compute_fid_score(sample_generator=sample(tf))
    
    print('FID scores:\n\t{}'.format('\n\t'.join(f'{k}: {v:0.1f}' for k, v in fid.items())))
    
    out_file = os.path.join(out_dir, 'fid.pkl')
    with open(out_file, 'wb') as f:
      pickle.dump(fid, f)

  sample_size = 8
  n_steps = 10 * sample_nrow
  ztest = zdist.sample((sample_size,))
  idx_modified = torch.randint(0, config['generator']['n_prim'], (sample_size,))
  vis_attr = ('vis_img_rgb', 'vis_layers_alpha')
  as_row_image = True
  as_gif = True
  
  def save_imdict(im_dict, out_dir):
    for vis_attr_i in vis_attr:
      imgs = im_dict[vis_attr_i]
      assert imgs.shape[1] == 1, f'Number of objects must be 1 but it is {imgs.shape[1]} for {vis_attr_i}.'
      imgs = imgs.view(sample_size, n_steps, *imgs.shape[2:])  # (BxN_tf)x1x... -> BxN_tfx...
    
      prefix = os.path.join(out_dir, vis_attr_i)
      save_tensor_images(imgs, prefix, nrow=sample_nrow, as_row_image=as_row_image, as_gif=as_gif)
  
  def draw_boxes(im_dict, idx_modified):
    idx_modified = idx_modified.repeat(n_steps, 1).transpose(0, 1).flatten(0, 1)
    n_total = len(idx_modified)
    for vis_attr_i in vis_attr:
      imgs = im_dict[vis_attr_i]
      assert imgs.shape[1] == 1, f'Number of objects must be 1 but it is {imgs.shape[1]} for {vis_attr_i}.'
      imgs = imgs.squeeze(1)        # (BxN_tf)x1x... -> (BxN_tf)x...
      masks = im_dict['layers_mask'][:, :, 0]      # these are color depth maps, so take only first channel
      
      # convert r-value of color map to binary mask
      masks = masks == 1
      
      # select masks of modified objects
      masks = masks[torch.arange(0, n_total), idx_modified]
      imgs = draw_box_batch(imgs, masks)
      im_dict[vis_attr_i] = imgs.view_as(im_dict[vis_attr_i])
  
  if 'trans' in eval_attr:
    print('Plot translation...')
    t_range = (-0.8, 0.8)
    
    for axis in ('x', 'y', 'z'):
      out_dir_trans = os.path.join(out_dir, 'translation', axis)
      os.makedirs(out_dir_trans, exist_ok=True)
      
      transform = ObjectTranslation(axis=axis)
      im_dict = evaluator.sample_transformed(ztest, transform, t_range, n_steps=n_steps, idx_modified=idx_modified)
      draw_boxes(im_dict, idx_modified)
      save_imdict(im_dict, out_dir_trans)

  if 'rot' in eval_attr:
    print('Plot rotation...')
    out_dir_rot = os.path.join(out_dir, 'rotation')
    os.makedirs(out_dir_rot, exist_ok=True)

    r_range = torch.tensor([0, 360.]) * math.pi / 180.
    transform = ObjectRotation(axis='y')
    im_dict = evaluator.sample_transformed(ztest, transform, r_range, n_steps=n_steps, idx_modified=idx_modified)
    draw_boxes(im_dict, idx_modified)
    save_imdict(im_dict, out_dir_rot)

  if 'cam' in eval_attr:
    print('Plot camera rotation...')
    out_dir_cam = os.path.join(out_dir, 'cam_rotation', 'azimuth')
    os.makedirs(out_dir_cam, exist_ok=True)
   
    azimuth = (0., 2.)
    polar = 0.25
    transform = CameraTransform(camera=generator.module.camera, uv=torch.tensor([azimuth[0], polar]), axis='x') #TODO: This was disabled
    im_dict = evaluator.sample_transformed(ztest, transform, azimuth, n_steps=n_steps)
    save_imdict(im_dict, out_dir_cam)

    out_dir_cam = os.path.join(out_dir, 'cam_rotation', 'polar')
    os.makedirs(out_dir_cam, exist_ok=True)
    
    azimuth = 0
    polar = (0.1, 0.5)
    transform = CameraTransform(camera=generator.module.camera, uv=torch.tensor([azimuth, polar[0]]), axis='y') #TODO: This was disabled
    im_dict = evaluator.sample_transformed(ztest, transform, polar, n_steps=n_steps)
    save_imdict(im_dict, out_dir_cam)
