import torch
import torch.nn.functional as F
from torchvision import models
import numpy as np
from scipy import linalg
from scipy.stats import entropy
import logging
from gan_training.metrics.inception import InceptionV3
from tqdm import tqdm

logger = logging.getLogger(__name__)


class InceptionEvaluator:
  def __init__(self, inception_nsplits=10, device=None, feat_dims=2048):
    self.mu_target = None
    self.sigma_target = None
    self.inception_nsplits = inception_nsplits
    self.device = device
    self.feat_dims = feat_dims
    
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[feat_dims]
    self.model = InceptionV3([block_idx]).to(self.device)
  
  def initialize_target(self, target_loader, total=None):
    logger.info('Initializing target statistics.')
    _, features = self.get_activations(target_loader, total=total)
    mu, sigma = self.get_statistics(features)
    self.mu_target = mu
    self.sigma_target = sigma
  
  def initialized(self):
    if self.mu_target is not None and self.sigma_target is not None:
      return True
    else:
      return False
  
  def evaluate_samples(self, samples, total=None):
    eval_dict = {}
    probabilities, features = self.get_activations(samples, total=total)
    
    if probabilities.ndim == 3:  # multiple outputs per sampled batch
      eval_dict['inception_i'] = []
      eval_dict['inception_i_std'] = []
      eval_dict['frechet_i'] = []
      for probabilities_i, features_i in zip(probabilities, features):
        inception, inception_std = self.get_inception_score(probabilities_i)
        eval_dict['inception_i'].append(inception)
        eval_dict['inception_i_std'].append(inception_std)
        
        if self.mu_target is not None and self.sigma_target is not None:
          mu, sigma = self.get_statistics(features_i)
          frechet = self.calculate_frechet_distance(
            mu, sigma, self.mu_target, self.sigma_target
          )
          eval_dict['frechet_i'].append(frechet)
      
      # for evaluation of all outputs together
      probabilities = np.concatenate(probabilities)
      features = np.concatenate(features)
    
    inception, inception_std = self.get_inception_score(probabilities)
    eval_dict['inception'] = inception
    eval_dict['inception_std'] = inception_std
    
    if self.mu_target is not None and self.sigma_target is not None:
      mu, sigma = self.get_statistics(features)
      frechet = self.calculate_frechet_distance(
        mu, sigma, self.mu_target, self.sigma_target
      )
      eval_dict['frechet'] = frechet
    
    return eval_dict
  
  def get_statistics(self, features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma
  
  def get_activations(self, samples, total=None):
    probabilities = []
    features = []
    
    cnt = 0
    for batch in tqdm(samples, total=total):
      if isinstance(batch, list):
        batch, _, _ = batch
      if not isinstance(batch, tuple):
        batch = [batch]
      
      probabilities_batch = []
      features_batch = []
      for output_i in batch:
        if output_i.ndimension() <= 1:  # in case of label
          continue
        output_i = output_i.to(self.device)[:, :3]  # slice in case of RGBD
        batch_size = output_i.shape[0]
        probabilities_i, features_i = self.get_activations_batch(output_i)
        features_i = features_i.view(batch_size, -1)
        probabilities_batch.append(probabilities_i.cpu().numpy())
        features_batch.append(features_i.cpu().numpy())
      
      probabilities.extend(np.stack(probabilities_batch, axis=1))
      features.extend(np.stack(features_batch, axis=1))
      
      cnt += 1
      if cnt == total:
        break
    
    probabilities = np.stack(probabilities).swapaxes(0, 1)
    features = np.stack(features).swapaxes(0, 1)
    
    if probabilities.shape[0] == 1:
      probabilities = probabilities[0]
      features = features[0]
    
    return probabilities, features
  
  def get_activations_batch(self, batch):
    self.model.eval()
    
    with torch.no_grad():
      logits, features = self.model(batch)
      probabilities = F.softmax(logits, dim=-1)
    
    # Assume that we only have one feature map
    features = features[0]
    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if features.shape[2] != 1 or features.shape[3] != 1:
      features = F.adaptive_avg_pool2d(features, output_size=(1, 1))
    
    return probabilities, features
  
  def get_inception_score(self, probabilities):
    N = probabilities.shape[0]
    splits = self.inception_nsplits
    # Now compute the mean kl-div
    split_scores = []
    
    for k in range(splits):
      part = probabilities[k * (N // splits): (k + 1) * (N // splits), :]
      py = np.mean(part, axis=0)
      scores = []
      for i in range(part.shape[0]):
        pyx = part[i, :]
        scores.append(entropy(pyx, py))
      split_scores.append(np.exp(np.mean(scores)))
    
    return np.mean(split_scores), np.std(split_scores)
  
  def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
            inception net (like returned by the function 'get_predictions')
            for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
            representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
            representative data set.

    Returns:
    --   : The Frechet Distance.
    """
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, \
      'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
      'Training and test covariances have different dimensions'
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
      msg = ('fid calculation produces singular product; '
             'adding %s to diagonal of cov estimates') % eps
      print(msg)
      offset = np.eye(sigma1.shape[0]) * eps
      covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
      if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
        m = np.max(np.abs(covmean.imag))
        raise ValueError('Imaginary component {}'.format(m))
      covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
