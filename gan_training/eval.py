import torch
from gan_training.distributions import interpolate_sphere
from gan_training.metrics import inception_score
from gan_training.metrics import FIDEvaluator
from torchvision.utils import make_grid
from utils.visualization import color_depth_maps, visualize_scenes, visualize_objects

class Evaluator(object):
  def __init__(self, generator, zdist, batch_size=64, device=None, single=False):
    self.generator = generator
    self.zdist = zdist
    self.batch_size = batch_size
    self.device = device
    self.single = single

    self.inception_eval = FIDEvaluator(
      device=device, batch_size=batch_size,
      resize=True
    )
  
  def compute_fid_score(self, sample_generator=None):
    if sample_generator is None:
      def sample():
        while True:
          z = self.zdist.sample((self.batch_size,))
          x = self.create_samples(z, param_transforms=[]) # disable param transforms for fid 
          if not self.single:
            rgb = x['img'][:, 0].cpu()
          else:
            rgb = x['img_single'].flatten(0,1).cpu()
          del x, z
          yield rgb
      
      sample_generator = sample()
    
    fid = self.inception_eval.get_fid(sample_generator)
    return fid
  
  def format_image_output(self, x):
    objects = visualize_objects(x, self.generator.module)
    scenes = visualize_scenes(x, self.generator.module)
    
    x.update(objects)
    x.update(scenes)
    
    n_tf = self.generator.module.n_tf
    n_fg = self.generator.module.n_fg
    bs = x['R'].shape[0] // (n_fg * n_tf)

    images = {}
    for k in x.keys():
      v = x[k]
      
      if v is None or v.ndim != 4: continue
      
      # use 3 channels
      n_ch = v.shape[1]
      if n_ch == 1:       # (BxNxN_tf)x1x... -> (BxNxN_tf)x3x...
        v = color_depth_maps(v[:, 0], rescale=False)[0] * 2 - 1
      v = v[:, :3]

      # (BxNxN_tf)x... -> (BxN_tf)xNx...
      v = v.view(bs, -1, n_tf, *v.shape[1:]).transpose(1, 2).flatten(0,1)
      images[k] = v
    
    return images
    
  def to_image(self, x, attr, nrow=8, nblock=2):
    n_fg = self.generator.module.n_fg

    scenes = []
    objects = []
    for k in attr:
      v = x[k]
      if v.shape[1] == 1:
        scenes.append(v)
      elif v.shape[1] == n_fg:
        v = v.split(1, dim=1)
        objects.append(v)
      else:
        raise RuntimeError(f'Cannot convert {k} with shape {v.shape} to image. Unknown number of objects.')
    
    # for each single object group all attributes together
    images = scenes
    for obj_imgs in zip(*objects):
      images.extend(obj_imgs)
    
    # organize into image blocks with nrow columns
    for i, img in enumerate(images):
      img = img.flatten(0, 1)[:, :3]      # (BxN_tf)x1xC... ->(BxN_tf)xC...
      if img.shape[1] == 1:
        img = color_depth_maps(img[:, 0], rescale=False)[0] * 2 - 1
      images[i] = img.split(nrow)
    
    image_blocks = []
    for imgs in zip(*images):
      image_blocks.extend(imgs)
      
    blocksize = len(images)
    img = torch.cat(image_blocks[:nblock*blocksize])
    return make_grid(img, nrow=nrow)

  def create_samples(self, z, param_transforms=None, obj_masks=None):
    self.generator.eval()

    param_transforms_orig = self.generator.module.param_transforms
    self.generator.module.param_transforms = param_transforms
    self.generator.module.n_tf = len(param_transforms) + 1
    
    with torch.no_grad():
      x = self.generator.module.forward(z, obj_masks=obj_masks)
      
    out = self.format_image_output(x)
    self.generator.module.param_transforms = param_transforms_orig
    self.generator.module.n_tf = len(param_transforms_orig) + 1
    return out

