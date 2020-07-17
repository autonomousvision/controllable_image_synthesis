import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools


class SizeLoss(nn.Module):
  """Compactness Loss"""
  def __init__(self):
    super(SizeLoss, self).__init__()
  
  def forward(self, x_out):
    # no gradient if area is relatively small
    area = x_out['prim_area'].clamp(min=0.1)
    return torch.mean(area)


class UVTransformer(nn.Module):
  """Transform image coordinates (u, v) to another camera viewpoint."""
  def __init__(self, K, imsize):
    super(UVTransformer, self).__init__()
    self.K = K.view(1, 3, 3)
    Ki = K.inverse()
    self.H, self.W = imsize

    u, v = np.meshgrid(range(self.H), range(self.W))
    uv = np.stack((u, v, np.ones_like(u)), axis=2)
    uv = torch.from_numpy(uv).float()
    uv = uv.flatten(0,1)          # HxWx3 -> (HxW)x3
    
    ray = uv.to(Ki.device) @ Ki.t()
    self.ray = ray.view(1, -1, 3)

  def transform(self, xyz, R=None, t=None):
    if t is not None:
      bs = xyz.shape[0]
      xyz = xyz - t.view(bs, 1, 3)
    if R is not None:
      xyz = torch.bmm(xyz, R)
    return xyz

  def unproject(self, depth, R=None, t=None):
    bs = depth.shape[0]
    
    ray = self.ray.to(depth.device)
    xyz = depth.view(bs, -1, 1) * ray
    
    xyz = self.transform(xyz, R, t)
    return xyz

  def project(self, xyz, R, t):
    bs = xyz.shape[0]
  
    xyz = torch.bmm(xyz, R.transpose(1, 2))
    xyz = xyz + t.reshape(bs, 1, 3)
  
    Kt = self.K.transpose(1, 2).expand(bs, -1, -1).to(xyz.device)
    uv = torch.bmm(xyz, Kt)
  
    d = uv[:, :, 2:3]
    uv = uv[:, :, :2] / (torch.nn.functional.relu(d) + 1e-12)
    return uv, d

  def forward(self, depth0, R0, t0, R1, t1):
    xyz = self.unproject(depth0, R0, t0)
    return self.project(xyz, R1, t1)


class RGBDConsistency(nn.Module):
  """Geometric Consistency Loss in one direction"""
  def __init__(self, K, imsize, clamp=0.):
    super().__init__()
    self.H, self.W = imsize
    self.clamp = clamp
    self.uv_transformer = UVTransformer(K, imsize)
    self.loss_fn = torch.nn.L1Loss(reduction='none')

  def forward_single(self, depth0, depth1, R0, t0, R1, t1, rgb0, rgb1, mask0, mask1, direction):
    bs = rgb0.shape[0]
    uv1, d1 = self.uv_transformer(depth0, R0, t0, R1, t1)
    
    # normalize [0, 1], then center around zero [-1, 1]
    uv1[..., 0] = uv1[..., 0] / (self.W - 1) * 2 - 1
    uv1[..., 1] = uv1[..., 1] / (self.H - 1) * 2 - 1
    uv1 = uv1.view(-1, self.H, self.W, 2)
    
    depth10 = torch.nn.functional.grid_sample(depth1, uv1, padding_mode='border')
    rgb10 = torch.nn.functional.grid_sample(rgb1, uv1, padding_mode='border')
    
    loss_d = self.loss_fn(d1.view(-1), depth10.view(-1))
    loss_rgb = self.loss_fn(rgb0.view(bs, 3, -1), rgb10.view(bs, 3, -1))
    loss_rgb = torch.mean(loss_rgb, dim=1).view(-1)
    
    mask0 = mask0.type_as(loss_d).view(-1)
    loss_d = loss_d * mask0
    loss_rgb = loss_rgb * mask0
    
    if self.clamp > 0:
      mask_clamp = (loss_d < self.clamp).float().detach()
      loss_d = loss_d * mask_clamp
      loss_rgb = loss_rgb * mask_clamp
    
    return loss_d.view(bs, -1).mean(dim=1), loss_rgb.view(bs, -1).mean(dim=1)
  
  def forward(self, depth0, depth1, R0, t0, R1, t1, rgb0, rgb1, mask0, mask1):
    l0, l0_rgb = self.forward_single(depth0, depth1, R0, t0, R1, t1, rgb0, rgb1, mask0, mask1, 0)
    l1, l1_rgb = self.forward_single(depth1, depth0, R1, t1, R0, t0, rgb1, rgb0, mask1, mask0, 1)
    return l0 + l1, l0_rgb + l1_rgb


class ConsistencyLoss(nn.Module):
  """Geometric Consistency Loss"""
  def __init__(self, n_tf, w_depth=1., w_rgb=1., max_batchsize=96, **kwargs):
    super(ConsistencyLoss, self).__init__()
    self.n_tf = n_tf
    self.w_d = w_depth
    self.w_rgb = w_rgb
    self.max_batchsize = max_batchsize
    
    self.loss_fn = RGBDConsistency(**kwargs)
    
  def _split_transforms(self, x_out):
    x = [{} for _ in range(self.n_tf)]
    for k, v in x_out.items():
      if v is None: continue
      v = v.view(-1, self.n_tf, *v.shape[1:])         # (BxNxN_tf)x... -> (BxN)xN_tfx...
      v = v.split(1, dim=1)
      for x_i, v_i in zip(x, v):
        x_i[k] = v_i.squeeze(1)
    
    x_tf = x[1:]
    x = x[0]
    return x, x_tf
  
  def _x_to_input(self, x, idx):
    d = x['layers_depth_rescaled'][idx]
    RT = x['prim_RT'][idx]
    R = RT[..., :3]
    t = RT[..., 3]
    rgb = x['layers_rgb'][idx] / 2 + 0.5
    mask = x['layers_alpha'][idx] / 2 + 0.5
    rgb = rgb * mask
    mask = (mask > 0.4).float()  # use a binary mask here
    return d, R, t, rgb, mask
  
  def forward(self, x_out):
    x, x_tf = self._split_transforms(x_out)
    perm = itertools.product([x], x_tf)

    loss_d = []
    loss_rgb = []
    batchsize = x['layers_depth_rescaled'].shape[0]

    if batchsize > self.max_batchsize:        # randomly select samples to avoid running out of memory
        idx = torch.randperm(batchsize)[0:self.max_batchsize]
    else:
        idx = torch.arange(batchsize).long().to(x['layers_depth_rescaled'].device)
    
    for x0, x1 in perm:
      d0, R0, t0, rgb0, mask0 = self._x_to_input(x0, idx)
      d1, R1, t1, rgb1, mask1 = self._x_to_input(x1, idx)

      loss_d_i, loss_rgb_i = self.loss_fn(d0, d1, R0, t0, R1, t1, rgb0, rgb1, mask0, mask1)
      
      loss_d.append(loss_d_i)
      loss_rgb.append(loss_rgb_i)
    
    loss_d = torch.cat(loss_d).mean()
    loss_rgb = torch.cat(loss_rgb).mean()
    return self.w_d * loss_d + self.w_rgb * loss_rgb

