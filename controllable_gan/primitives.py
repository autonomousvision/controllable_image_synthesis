import neural_renderer as nr

class Primitive:
  """Primitive configurations and templates"""
  def __init__(self):
    self.v = None
    self.f = None

  def __len__(self):
    raise ValueError('Needs to be implemented!')

class PointCloud(Primitive):
  """Point cloud configurations"""
  def __init__(self, n_points, n_channel, transform_feat=False):
    super().__init__()
    self.n_points = n_points
    self.n_channel = n_channel
    self.transform_feat=transform_feat

  def __len__(self):
    return self.n_points * (3 + self.n_channel)

class Cuboid(Primitive):
  """Cuboid configurations"""
  def __init__(self, texsize, n_channel):
    super().__init__()
    self.texsize = texsize
    self.n_channel = n_channel

  def __len__(self):
    return 6 * self.texsize**2 * self.n_channel

class Mesh(Primitive):
  """Mesh templates"""
  def __init__(self, render_type, n_channel):
    super().__init__()
    primitive_type = render_type
    self.n_channel = n_channel
    if primitive_type == 'sphere':
        v, f = nr.load_obj('controllable_gan/templates/sphere_114.obj')
    elif primitive_type == 'cuboid':
        v, f = nr.load_obj('controllable_gan/templates/cube.obj', False)
    else:
      raise AttributeError
    self.n_latent = 128
    self.v = v
    self.f = f

  def __len__(self):
    return self.n_latent 

