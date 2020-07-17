import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class Generator2D(nn.Module):
  """2D Generator that transforms projected feature to photorealistic images"""

  def __init__(self, input_nc, output_nc, nlayers=4, nfilter=128, nfilter_max=512, 
                     resnet=True, norm="adain", down_sample=1, zdim=256, n_prim=1):
    """2D Generator initialization
    
    Args:
        input_nc (int): Number of input channels 
        output_nc (int): Number of output channels 
        nlayers (int, optional): Number of unit blocks (Resnet blocks or plain convolutions). Defaults to 4.
        nfilter (int, optional): Number of convolutional filters. Defaults to 128.
        nfilter_max (int, optional): Maximum number of convolutional filters. Defaults to 512.
        resnet (bool, optional): If true then use Resnet blocks else plain convolutions. Defaults to True.
        norm (str, optional): Normalization method ("bn", "in", "adain"). Defaults to "adain".
        down_sample (int, optional): Number of downsampling/upsampling pairs. Defaults to 1.
        zdim (int, optional): Dimension of the noise vector, needed for adain. Defaults to 256.
        n_prim (int, optional): Number of primitives, needed for adain. Defaults to 1.
    """
    
    super().__init__()

    input_size = 64
    s0 = self.s0 = int(input_size/ 2**nlayers)
    nf = self.nf = nfilter
    nf_max = self.nf_max = nfilter_max
    self.n_prim = n_prim

    if norm=="bn":
      norm_layer = nn.BatchNorm2d
    elif norm == "in":
      norm_layer = nn.InstanceNorm2d
    elif norm == "adain":
      norm_layer = AdaptiveInstanceNorm2d 
    else:
      raise ValueError("Unknown normalization method!")
    self.norm = norm

    # Submodules
    nfilter_init = 128
    self.nf0 = min(nf_max, nf * 2**nlayers)

    if down_sample>0:
        assert (nlayers%2==0)
        assert (down_sample < nlayers/2)

    self.conv_img_in = nn.Conv2d(input_nc, nfilter, 3, padding=1)

    # build blocks
    blocks = []
    nf0 = nfilter 
    nf1 = nf1_init = nfilter 
    for i in range(nlayers):
      if i>0:
        nf0 = nf1
      if resnet:
        blocks += [
            ResnetBlock(nf0, nf1, norm_layer=norm_layer),
        ]
      else:
        blocks += [
            nn.Conv2d(nf0, nf1, 3, stride=1, padding=1),
            norm_layer(nf1),
            nn.LeakyReLU(0.2),
        ]
      if down_sample>0: 
        if i<down_sample:
          nf1_init = nf1
          nf1 = min(nf1*2, nfilter_max)
          blocks += [
              nn.Conv2d(nf1_init, nf1, 4, stride=2, padding=1),
              nn.BatchNorm2d(nf1),
              nn.LeakyReLU(0.2),
              #nn.AvgPool2d(3, stride=2, padding=1),
          ]
        elif i>=nlayers/2 and i<nlayers-1:
          nf1_init = nf1
          nf1 = max(nf1//2, nfilter_init)
          blocks += [
              nn.ConvTranspose2d(nf1_init, nf1, 4, stride=2, padding=1),
              nn.BatchNorm2d(nf1),
              nn.LeakyReLU(0.2),
              #nn.Upsample(scale_factor=2, mode='bilinear'),
          ]

    self.resnet = nn.Sequential(*blocks)
    self.conv_img = nn.Conv2d(nf1, output_nc, 3, padding=1)

    # Initiate mlp (predicts AdaIN parameters)
    if self.norm=='adain':
        num_adain_params = self.get_num_adain_params()
        num_adain_params = num_adain_params*self.n_prim # apply different normalization to each primitive
        self.mlp = MLP(zdim, num_adain_params)

  def get_num_adain_params(self):
    """Return the number of AdaIN parameters needed by the model"""
    num_adain_params = 0
    for m in self.modules():
      if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
        num_adain_params += 2 * m.num_features
    return num_adain_params

  def assign_adain_params(self, adain_params):
    """Assign the adain_params to the AdaIN layers in model"""
    # Use the same bias and weight across different objects, not sure if it is good
    if adain_params.ndim == 2:
      adain_params = adain_params.view(adain_params.shape[0], self.n_prim, -1)
      #adain_params = adain_params.repeat(self.n_prim, 1, 1).transpose(0, 1)
    
    for m in self.modules():
      if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
        # Extract mean and std predictions
        mean = adain_params[:, :, :m.num_features]
        std = adain_params[:, :, m.num_features:2*m.num_features]
        
        # Update bias and weight
        m.bias = mean.contiguous().view(-1)
        m.weight = std.contiguous().view(-1)
        
        # Move pointer
        if adain_params.size(2) > 2 * m.num_features:
          adain_params = adain_params[:, :, 2*m.num_features:]

  def forward(self, im, z=None):
    """Forward function
    
    Args:
        im (torch.FloatTensor): Images from differentiable renderer 
        z ([type], optional): Noise vector, needed for AdaIN. Defaults to None.
    
    Returns:
        out (torch.FloatTensor): Predicted photorealistic image and alpha map
    """
    
    batch_size = im.shape[0]
    max_batch_size = (128 // self.n_prim) * self.n_prim       # ensure not to split objects from same scene, as they share same adain parameters
    im = self.conv_img_in(im)
    if batch_size>max_batch_size:
        n_split = np.ceil(float(batch_size)/max_batch_size).astype(np.int)
        out = []
        for i in range(n_split):
          # Update AdaIN parameters by MLP prediction based off style code
          start, end = i*max_batch_size, (i+1)*max_batch_size
          if self.norm == 'adain':
            self.assign_adain_params(self.mlp(z[start//self.n_prim:end//self.n_prim]))
          out_i = self.resnet(im[start:end])
          out_i = self.conv_img(actvn(out_i))
          out.append(out_i)
        out = torch.cat(out, dim=0)
    else:
        if self.norm == 'adain':
          self.assign_adain_params(self.mlp(z))
        out = self.resnet(im)
        out = self.conv_img(actvn(out))
    out = torch.tanh(out)

    return out

class MLP(nn.Module):
    """Multilayer Perceptron Network"""
    def __init__(self, input_dim, output_dim, dim=256, n_blk=3, activ="relu"):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, dim), nn.ReLU(inplace=True)]
        for _ in range(n_blk - 2):
            layers += [nn.Linear(dim, dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

class AdaptiveInstanceNorm2d(nn.Module):
    """Adaptive Instance Normalization

    Reference: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/munit/models.py
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c, h, w = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, h, w)

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps
        )

        return out.view(b, c, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"

class ResnetBlock(nn.Module):
    """Resnet Block
    
    Reference: https://github.com/LMescheder/GAN_stability/blob/master/gan_training/models/resnet.py 
    """
    def __init__(self, fin, fout, fhidden=None, is_bias=True, norm_layer=None):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

        if norm_layer is not None:
            self.conv_0 = nn.Sequential(*[self.conv_0, norm_layer(self.fhidden)])
            self.conv_1 = nn.Sequential(*[self.conv_1, norm_layer(self.fout)])


    def forward(self, x):
        x_s = self._shortcut(x)
        #x_s = self.block(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out


