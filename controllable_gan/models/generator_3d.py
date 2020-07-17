import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import neural_renderer as nr
from utils.commons import get_rotation_from_two_vecs, \
        get_spherical_coords, compute_uvsampler
        

class Generator3D(nn.Module):
  """3D Generator that generates a set of primitives as well as a background sphere """
  def __init__(self, z_dim, n_prim, primitive, bg_cube, n_hidden_bg=128, template_bg='controllable_gan/templates/sphere_642.obj'):
    """3D Generator initialization
    
    Args:
        z_dim (int): Dimension of the noise vector 
        n_prim (int): Number of foreground primitives
        primitive (Primitive): Primitive class with template and configurations 
        bg_cube (bool): Use background sphere or not 
        n_hidden_bg (int, optional): Dimension of the latent space of the background feature generation branch. Defaults to 128.
        template_bg (str, optional): Path to the background sphere template. Defaults to 'controllable_gan/templates/sphere_642.obj'.
    """

    super(Generator3D, self).__init__()

    n_tr = 6 + 3 + 3
    n_feat = len(primitive)

    n_hidden = [128, 256, 512]

    # generate the 3d parameters for each object jointly
    self.n_prim = n_prim
    self.bg_cube = bg_cube
    self.n_hidden_bg = n_hidden_bg
    self.primitive = primitive

    self.z_dim_half = z_dim_half = z_dim // 2

    # constants
    self.prim_size_max = 0.6
    self.prim_size_min = 0.1

    # transformations
    mlp_tr = [nn.Linear(z_dim_half, n_hidden[0]),
              nn.LeakyReLU(0.2),
              nn.Linear(n_hidden[0], n_hidden[1]),
              nn.LeakyReLU(0.2),
              nn.Linear(n_hidden[1], n_hidden[2]),
              nn.LeakyReLU(0.2),
              nn.Linear(n_hidden[2], n_tr*n_prim)]
    self.mlp_tr = nn.Sequential(*mlp_tr)

    # features 
    mlp_feat = [nn.Linear(z_dim_half, n_hidden[0]),
                nn.LeakyReLU(0.2),
                nn.Linear(n_hidden[0], n_hidden[1]),
                nn.LeakyReLU(0.2),
                nn.Linear(n_hidden[1], n_hidden[2]),
                nn.LeakyReLU(0.2),
                nn.Linear(n_hidden[2], n_feat*n_prim)]
    self.mlp_feat = nn.Sequential(*mlp_feat)
    if primitive.v is not None and primitive.f is not None:
      self.texture_predictor = self.get_texture_predictor(n_feat, primitive.v, primitive.f, texture_channel=primitive.n_channel)

    # background sphere
    if bg_cube:
        self.mlp_bg = nn.Linear(z_dim_half, n_hidden_bg)
        v_bg, f_bg = nr.load_obj(template_bg)
        self.texture_predictor_bg = self.get_texture_predictor(n_hidden_bg, v_bg, f_bg, texture_channel=primitive.n_channel)
    self.tanh = nn.Tanh()
    self.sigmoid = nn.Sigmoid()

  def get_texture_predictor(self, n_feat, v, f, tex_size=4, texture_channel=3):
    """Predict 2D texture maps and map to mesh based on UV coordinates
    
    Args:
        n_feat (int): Dimension of the latent feature for predicting texture 
        v (torch.FloatTensor): Vertices of the mesh 
        f (torch.FloatTensor): Faces of the mesh 
        tex_size (int, optional): Texture resolution. Defaults to 4.
        texture_channel (int, optional): Number of channels of the texture map. Defaults to 3.
    
    Returns:
        texture_predictor: A network to predict the texture map
    """
    n_f = f.shape[0]

    uv_sampler = compute_uvsampler(v.cpu().numpy(), f.cpu().numpy(), tex_size=tex_size)
    # F' x T x T x 2
    uv_sampler = torch.from_numpy(uv_sampler).float()
    # B x F' x T x T x 2
    uv_sampler = uv_sampler.unsqueeze(0) #.repeat(self.opts.batch_size, 1, 1, 1, 1)
    img_H = int(2**np.floor(np.log2(np.sqrt(n_f) * tex_size)))
    img_W = 2 * img_H
    texture_predictor = TexturePredictorUV(
      n_feat, uv_sampler, img_H=img_H, img_W=img_W,
      texture_channel=texture_channel,
      upsampling_mode='transpose')

    return texture_predictor


  def param_to_pose(self, x):
    """Tranfrom the variables to the space of pose parameters
    
    Args:
        x (torch.FloatTensor): Prediction of the 3D generator, BxNx12 
    
    Returns:
        x (dict): Pose parameters {'R', 't', 'scale'} 
    """
    bs = x.shape[0]

    x = x.view(bs * self.n_prim, 12)

    rotation_vec = x[:, 0:6]
    rotation_mat = get_rotation_from_two_vecs(rotation_vec.view(-1, 3, 2))
    rotation_mat = rotation_mat.view(-1, 3, 3)
    translation = self.tanh(x[:,6:9])
    scale = self.sigmoid(x[:,9:12]) * (self.prim_size_max-self.prim_size_min) + self.prim_size_min  # [0.1, 0.6]

    x = {'R': rotation_mat,
         't': translation,
         'scale': scale}

    return x


  def forward(self, x):
    """Predict a set of 3D primitives
    
    Args:
        x (torch.FloatTensor): Input noise vector 
    
    Returns:
        out (dict): Primitive pose parameters and feature paramters {'R', 't', 'scale', 'feature', 'bg'} 
    """
    bs = x.shape[0]

    # transformations
    x_transform = self.mlp_tr(x[:,:self.z_dim_half])
    out = self.param_to_pose(x_transform)

    # features
    x_feature = self.mlp_feat(x[:,self.z_dim_half:])
    if self.primitive.v is not None and self.primitive.f is not None:
        x_feature = self.texture_predictor(x_feature.view(bs*self.n_prim, -1))
    out['feature'] = x_feature.view(bs * self.n_prim, -1, *x_feature.shape[2:])
    
    # background sphere
    if self.bg_cube:
        x_bg = self.mlp_bg(x[:,self.z_dim_half:])
        x_feature_bg = self.texture_predictor_bg(x_bg)
        out['bg'] = x_feature_bg

    return out 

class TexturePredictorUV(nn.Module):
  """Texture map generator"""

  def __init__(self, nz_feat, uv_sampler, img_H=64, img_W=128, texture_channel=3, n_upconv=3, nc_init=64, num_sym_faces=624, upsampling_mode='transpose'):
    super(TexturePredictorUV, self).__init__()
    self.feat_H = img_H // (2 ** n_upconv)
    self.feat_W = img_W // (2 ** n_upconv)
    self.nc_init = nc_init
    self.num_sym_faces = num_sym_faces
    self.F = uv_sampler.size(1)
    self.T = uv_sampler.size(2)
    # B x F x T x T x 2 --> B x F x T*T x 2
    self.uv_sampler = uv_sampler.view(-1, self.F, self.T*self.T, 2)

    self.enc = nn.Linear(nz_feat, self.nc_init*self.feat_H*self.feat_W)
    nc_final = texture_channel

    blocks = []
    nf_max = 512
    for i in range(n_upconv):
      nf0 = min(nc_init, nf_max)
      nf1 = min(nc_init, nf_max)
      if upsampling_mode=='transpose':
          blocks += [nn.ConvTranspose2d(nf0, nf1,
                                       kernel_size=3, stride=2,
                                       padding=1, output_padding=1),
                    nn.BatchNorm2d(nf1),
                    nn.LeakyReLU(0.2, True)]
      elif upsampling_mode=='nearest' or upsampling_mode=='bilinear':
          blocks += [
              nn.Conv2d(nf0, nf1, 3, padding=1),
              nn.BatchNorm2d(nf1),
              nn.LeakyReLU(0.2, True),
              nn.Upsample(scale_factor=2, mode=upsampling_mode)
          ]

    blocks += [
        nn.Conv2d(nf1, nc_final, 3, padding=1),
        nn.BatchNorm2d(nc_final),
    ]
    self.decoder = nn.Sequential(*blocks)


  def forward(self, feat):

    uvimage_pred = self.enc.forward(feat)
    uvimage_pred = uvimage_pred.view(uvimage_pred.size(0), self.nc_init, self.feat_H, self.feat_W)
    # B x 2 or 3 x H x W
    self.uvimage_pred = self.decoder.forward(uvimage_pred)

    bs = uvimage_pred.shape[0]
    tex_pred = torch.nn.functional.grid_sample(self.uvimage_pred, self.uv_sampler.repeat(bs, 1, 1, 1).to(feat.device))
    tex_pred = tex_pred.view(uvimage_pred.size(0), -1, self.F, self.T, self.T).permute(0, 2, 3, 4, 1)

    # Contiguous Needed after the permute..
    return tex_pred.contiguous()

