"""
This files contains all utilities needed for read and write operations.
"""
import os
import yaml
import glob
import imageio
import math
import torch
from torchvision.utils import save_image


def make_gif(im_dir, out_file, pattern='*.png', fps=10):
  """
  Create .gif from given images.
  Args:
    im_dir (str): path to folder with the images
    out_file (str): path to folder with the images
    pattern (str): pattern for filtering files in im_dir
    fps (int): frames per second

  """
  im_files = glob.glob(os.path.join(im_dir, pattern))
  if len(im_files) == 0:
    raise ValueError(f'No images found in {im_dir}!')
  
  writer = imageio.get_writer(out_file, mode='I', fps=fps)
  for im_file in im_files:
    im = imageio.imread(im_file)
    writer.append_data(im)
  writer.close()


def save_tensor_images(images, prefix=None, nrow=None, as_row_image=True, as_gif=False):
  assert images.ndim == 5, f'Input has to be 5D but it is {images.ndim}D.'
  if prefix is None:
    prefix = ''
  else:
    prefix = prefix + '_'
  
  if nrow is None:
    nrow = images.shape[1]
  else:
    nrow = min(nrow, images.shape[1])
  
  for i in range(images.shape[0]):
    if as_row_image:
      filename = prefix + f'{i:04d}.png'
      stepsize = math.ceil(images.shape[1] / nrow)
      imgs = images[i, ::stepsize].clone()
      save_image(imgs, filename, nrow=nrow)
    if as_gif:
      filename = prefix + f'{i:04d}.gif'
      if os.path.isfile(filename):      # prevent appending to existing file
        os.remove(filename)
        
      with imageio.get_writer(filename, mode='I', fps=10) as writer:
        for j in range(images.shape[1]):
          # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
          ndarr = images[i, j].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
          writer.append_data(ndarr)


def save_frames(frames, out_dir, as_row=True, as_gif=False):
  """
  Save frames to a single row or as a gif.
  Args:
    frames (torch.FloatTensor or torch.ByteTensor): frames to save, N x N_columns x C x H x W
    out_dir: output directory
    as_row: save frames in a single row
    as_gif: concatenate columns to gif

  """
  os.makedirs(out_dir, exist_ok=True)
  if frames.dtype == torch.uint8:  # save_image needs float value in [0, 1]
    frames = frames.float()
    frames = frames / 255.
  if as_gif:
    gif_dir = 'gif_images'
    os.makedirs(os.path.join(out_dir, gif_dir), exist_ok=True)
  for i, frames_i in enumerate(frames):
    if as_row:
      out_file = os.path.join(out_dir, f'img_{i:04d}.png')
      save_image(frames_i.clone(), out_file, nrow=frames_i.shape[0])
    if as_gif:
      for j, frame in enumerate(frames_i):
        out_file = os.path.join(out_dir, gif_dir, f'img_{i:04d}_{j:04d}.png')
        save_image(frame.unsqueeze(0), out_file)
      
      out_file = os.path.join(out_dir, f'img_{i:04d}.gif')
      make_gif(os.path.join(out_dir, gif_dir), out_file, pattern=f'img_{i:04d}_*', fps=10)
  
  print(f'Saved images to {out_dir}')

