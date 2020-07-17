import argparse
import os
from os import path
import time
import torch
from torch import nn
from gan_training.train import Trainer
from gan_training.logger import Logger
from gan_training.checkpoints import CheckpointIO
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (
  load_config
)
from controllable_gan.config import get_dataloader, get_out_dir, build_models, build_optimizers, build_g_losses, save_config

torch.manual_seed(0)


if __name__ == '__main__':
  # Arguments
  parser = argparse.ArgumentParser(
    description='Train 3d controllable GAN.'
  )
  parser.add_argument('config', type=str, help='Path to config file.')
  args = parser.parse_args()
  
  config = load_config(args.config, 'configs/default.yaml')
  is_cuda = (torch.cuda.is_available())
  assert is_cuda, 'No GPU device detected!'
  
  # Short hands
  nlabels = config['data']['nlabels']
  batch_size = config['training']['batch_size']
  restart_every = config['training']['restart_every']
  fid_every = config['training']['fid_every']
  fid_single_every = config['training']['fid_single_every']
  save_every = config['training']['save_every']
  backup_every = config['training']['backup_every']
  
  out_dir = get_out_dir(config)
  checkpoint_dir = path.join(out_dir, 'chkpts')
  
  # Create missing directories
  if not path.exists(out_dir):
    os.makedirs(out_dir)
  if not path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  
  # Save config for this run
  save_config(os.path.join(out_dir, 'config.yaml'), config)
  
  # Logger
  checkpoint_io = CheckpointIO(
    checkpoint_dir=checkpoint_dir
  )
  
  device = torch.device("cuda:0" if is_cuda else "cpu")
  
  # Dataset
  train_loader = get_dataloader(config, split='train')
  
  # Create models
  generator, discriminator = build_models(config)
  print(generator)
  print(discriminator)
  
  # Put models on gpu if needed
  generator = generator.to(device)
  discriminator = discriminator.to(device)
  
  g_optimizer, d_optimizer = build_optimizers(
    generator, discriminator, config
  )
  
  # Use multiple GPUs if possible
  generator = nn.DataParallel(generator)
  discriminator = nn.DataParallel(discriminator)
  
  # Register modules to checkpoint
  checkpoint_io.register_modules(
    generator=generator,
    discriminator=discriminator,
    g_optimizer=g_optimizer,
    d_optimizer=d_optimizer,
  )
  
  # Get model file
  model_file = config['training']['model_file']
  
  # Logger
  logger = Logger(
    log_dir=path.join(out_dir, 'logs'),
    img_dir=path.join(out_dir, 'imgs'),
    monitoring=config['training']['monitoring'],
    monitoring_dir=path.join(out_dir, 'monitoring')
  )
  
  # Distributions
  ydist = get_ydist(nlabels, device=device)
  zdist = get_zdist('gauss', config['z_dist']['dim'],
                    device=device)
  
  # Save for tests
  x_real, _ = next(iter(train_loader))
  ztest = zdist.sample((batch_size,))
  
  # Test generator
  generator_test = generator
  
  # Evaluator
  evaluator = Evaluator(generator_test, zdist, batch_size=config['test']['batch_size'], device=device)
  evaluator_single = Evaluator(generator_test, zdist, batch_size=config['test']['batch_size'], device=device, single=True)
  
  # initialize fid evaluators
  if fid_every > 0:
    cache_file = os.path.join(out_dir, 'cache_train.npz')
    val_loader = get_dataloader(config, split='val')
    evaluator.inception_eval.initialize_target(val_loader, cache_file=cache_file)
  if fid_single_every > 0:
    cache_file = os.path.join(out_dir, 'cache_train_single.npz')
    val_loader_single = get_dataloader(config, split='val', single=True)
    evaluator_single.inception_eval.initialize_target(val_loader_single, cache_file=cache_file)
  
  # Train
  tstart = t0 = time.time()
  fid_best = float('inf')
  
  # Load checkpoint if it exists
  try:
    load_dict = checkpoint_io.load(model_file)
  except FileNotFoundError:
    it = epoch_idx = -1
  else:
    it = load_dict.get('it', -1)
    epoch_idx = load_dict.get('epoch_idx', -1)
    fid_best = load_dict.get('fid_best', float('inf'))
    logger.load_stats('stats.p')
  
  # Additional losses to GAN loss
  losses_g, losses_g_2d = build_g_losses(generator.module, config)

  # Trainer
  trainer = Trainer(
    generator, discriminator, g_optimizer, d_optimizer,
    'standard', 'real', 10.,
    losses_g=losses_g, losses_g_2d=losses_g_2d,
    n_labels=nlabels
  )
  
  # Training loop
  print('Start training...')
  while True:
    epoch_idx += 1
    print('Start epoch %d...' % epoch_idx)
    
    for x_real, y in train_loader:
      it += 1
      
      x_real, y = x_real.to(device), y.to(device)
      y.clamp_(None, nlabels - 1)
      
      losses = {}
      
      # Discriminator updates
      z = zdist.sample((batch_size,))
      dloss = trainer.discriminator_trainstep(x_real, y, z)
      losses.update(dloss)
      
      # Generators updates
      z = zdist.sample((batch_size,))
      gloss = trainer.generator_trainstep(y, z)
      losses.update(gloss)
      
      for name, val in losses.items():
        logger.add('losses', name, val, it=it)
      
      # Print stats
      print('[epoch {:0d}, it {:4d}] {}'.format(epoch_idx, it,
            ', '.join(f'{name} = {val:.4f}' for name, val in losses.items())))

      # (i) Sample if necessary
      if (it % config['training']['sample_every']) == 0:
        print('Creating samples...')
        
        x = evaluator.create_samples(ztest, generator.module.param_transforms)
        
        x['real'] = x_real[:, :3].repeat(generator.module.n_tf, 1, 1, 1, 1)\
          .transpose(0, 1).flatten(0, 1).unsqueeze(1)  # (BxN_tf)xCxHxW

        #attr = ['real', 'vis_img_rgb', 'vis_prim', 'vis_layers_alpha', 'layers_rgb', 'layers_alpha']
        attr = ['real', 'vis_img_rgb', 'vis_prim', 'vis_layers_alpha', 'layers_rgb']
        if config['generator']['bg_cube']:
          attr.insert(2, 'vis_img_bg_rgb')
        
        img = evaluator.to_image(x, attr, nrow=min(batch_size*generator.module.n_tf, 8), nblock=2)
        logger.add_imgs(img, 'rgb', it)
        
        # free memory
        del x, x_real, img
        
      # (ii) Compute fid if necessary
      if fid_every > 0 and ((it + 1) % fid_every) == 0:
        fid = evaluator.compute_fid_score()
        logger.add('fid_score', 'all', fid, it=it)
        torch.cuda.empty_cache()
        # save best model
        if fid < fid_best:
          fid_best = fid
          print('Saving best model...')
          checkpoint_io.save('model_best.pt', it=it)
          logger.save_stats('stats_best.p')
          torch.cuda.empty_cache()
        
      if fid_single_every > 0 and ((it + 1) % fid_single_every) == 0:
        fid = evaluator_single.compute_fid_score()
        logger.add('fid_score', 'single', fid, it=it)
        torch.cuda.empty_cache()

      # (iii) Backup if necessary
      if ((it + 1) % backup_every) == 0:
        print('Saving backup...')
        checkpoint_io.save('model_%08d.pt' % it, it=it)
        logger.save_stats('stats_%08d.p' % it)
        torch.cuda.empty_cache()
      
      # (iv) Save checkpoint if necessary
      if time.time() - t0 > save_every:
        print('Saving checkpoint...')
        checkpoint_io.save(model_file, it=it)
        logger.save_stats('stats.p')
        t0 = time.time()
        torch.cuda.empty_cache()

        if (restart_every > 0 and t0 - tstart > restart_every):
          exit(3)
