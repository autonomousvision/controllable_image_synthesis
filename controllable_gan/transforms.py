import torch
import math
from utils.commons import get_rotation_from_axis_angle


class Param3DTransform(torch.nn.Module):
  """Base class that defines layout of the parameter vector.
  Args:
    axis (str, optional): Axis along which to perform transform.
    Can be 'rand' to randomly choose an axis or any combination of ['x', 'y', 'z'], e.g. 'x' or 'xy'.
    param_range ((float, float), optional): Parameter range of transform, e.g. shift for translation, angle for rotation.

  """

  def __init__(self, axis='rand', param_range=(0, 1)):
    self.k_rot = 'R'
    self.k_trans = 't'

    self.axis = axis
    self.param_range = param_range

  def sample_axes(self, axis=None, N=1):
    """Randomly sample N axes
    Args:
            axis (str, optional): Axis along which to perform transform. If specified, use this instead of class attribute.
            N (int, optional): Number of sampled axes. Defaults to 1.

        Returns:
            torch.BoolTensor: Sampled axes, Nx3.

    """
    if axis is None:
      axis = self.axis

    if not any(['x' in axis, 'y' in axis, 'z' in axis]) and axis != 'rand':
      raise AttributeError('axis must be "rand" or contain "x", "y" or "z"')

    if axis == 'rand':
      idcs = torch.randint(0, 3, (N,))
      ax = torch.zeros((N, 3), dtype=torch.bool)
      ax[range(N), idcs] = 1
      return ax

    # randomly choose N objects from possible axes
    ax = []
    if 'x' in axis:
      ax.append(torch.BoolTensor([1, 0, 0]))
    if 'y' in axis:
      ax.append(torch.BoolTensor([0, 1, 0]))
    if 'z' in axis:
      ax.append(torch.BoolTensor([0, 0, 1]))

    ax = torch.stack(ax).repeat(math.ceil(N / len(ax)), 1)
    ax = ax[torch.randperm(len(ax))[:N]]
    return ax

  def sample_param(self, N):
    """Sample transformation parameter uniformly within parameter range."""
    param = torch.rand(N) * (self.param_range[1] - self.param_range[0]) + self.param_range[0]
    return param

  def __call__(self, x, value=None, axis=None, mask=None):
    """Apply transform to input.
      Args:
          x (dict): Parameter dict to be transformed.
          value (float): Value used in transform.
          axis (str, optional): Axis along which to perform transform. If specified, use this instead of class attribute.
          mask (torch.BoolTensor, optional): Mask for selecting primitives. Defaults to None.

      Returns:
          dict: Transformed parameter dict.
    """
    raise NotImplementedError


class ObjectRotation(Param3DTransform):
  def __init__(self, **kwargs):
    super(ObjectRotation, self).__init__(**kwargs)

  def __call__(self, x, value=None, axis=None, mask=None):
    """"Extrinsic rotation of the primitives.
    The rotation is performed relative to the current rotation of each primitive."""
    R0 = x[self.k_rot]
    if mask is None:  # transform all
      mask = torch.ones(len(R0), dtype=torch.bool)

    N = mask.sum().item()

    axis = self.sample_axes(axis, N)

    # sample rotations
    if value is not None:  # rotate objects for given angle
      angle = value * torch.ones(N)
    else:  # sample angle randomly
      angle = self.sample_param(N)

    R = torch.stack([get_rotation_from_axis_angle(axis_i, angle_i) for axis_i, angle_i in zip(axis.float(), angle)])

    # Apply R to current rotations R0
    mask = mask.flatten()     # BxN_obj -> (BxN_obj)
    R1 = R.type_as(R0).bmm(R0[mask])  # extrinsic rotation
    
    # Only modify primitives which are not masked out
    x[self.k_rot] = R0.masked_scatter(mask.view(-1, 1, 1).to(R0.device), R1)
    return x


class ObjectTranslation(Param3DTransform):
  def __init__(self, **kwargs):
    super(ObjectTranslation, self).__init__(**kwargs)

  def __call__(self, x, value=None, axis=None, mask=None):
    """Translation of the primitives.
    The translation defines the absolute position of each primitive."""
    t0 = x[self.k_trans]
    if mask is None:
      mask = torch.ones(len(t0), 1, dtype=torch.bool)

    N = mask.sum().item()

    axis = self.sample_axes(axis, N)

    # sample positions
    if value is not None:  # set position to given value
      t1 = value * torch.ones(N)
    else:  # sample positions randomly
      t1 = self.sample_param(N)

    # Only modify primitives which are not masked out at given axis
    axis = axis.unsqueeze(1).expand(-1, mask.shape[1], -1)    # Bx3 -> BxN_objx3
    mask = mask.unsqueeze(2).expand(-1, -1, 3)                # BxN_obj -> BxN_objx3
    mask = (mask & axis).flatten(0, 1)
    x[self.k_trans] = t0.masked_scatter(mask.to(t0.device), t1.to(t0.device))
    return x
