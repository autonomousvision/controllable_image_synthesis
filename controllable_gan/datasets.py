import os
import numpy as np
import torchvision
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import glob
from collections import Counter


class ObjectDataset(VisionDataset):
  """
  Multiple data directories for varying number of objects per scene.
  Folder structure: root/split/scene/scene_idx.png and
                    root/split/scene/scene_depthidx.npy and
                    root/split/scene/bboxes.npz
  """

  def __init__(self, data_dirs, split, transforms=None, nlabels=1):
    # Use multiple root folders
    if not isinstance(data_dirs, list):
      data_dirs = [data_dirs]
  
    # assign label for each root folder
    self.nlabels = nlabels
    if self.nlabels not in [1, 2]:
      raise NotImplementedError
    labels = [self._get_target(ddir) for ddir in data_dirs]
    data_dirs = [os.path.join(ddir, split) for ddir in data_dirs]
  
    if transforms is None:
      transforms = torchvision.transforms.ToTensor()  # HxWxC -> CxHxW
  
    # initialize base class
    super(ObjectDataset, self).__init__(root=data_dirs, transform=transforms)
    
    self.filenames = []
    self.labels = []
  
    for ddir, label in zip(self.root, labels):
      if label == 1 and self.nlabels == 1:    # do not add pure bg images
        continue
    
      filenames = self._get_filenames(ddir)
      self.filenames.extend(filenames)
      self.labels.extend([label] * len(filenames))
    
    labels = np.array(self.labels)
    if self.nlabels > 1 and split == 'train' and not np.any(labels == 1):
      raise ValueError('No background folder provided!')
  
    if nlabels > 1 and split == 'train':  # get equally many pure bg and bg+fg images
      make_equal_label(self.filenames, self.labels)

  def __len__(self):
    return len(self.filenames)

  @staticmethod
  def _get_filenames(root_dir):
    return glob.glob(f'{root_dir}/*/*.png')

  @staticmethod
  def _get_num_obj(path):
    return int(os.path.basename(path).split('_')[0][-1])

  def _get_target(self, path):
    """
    Args:
        path (string): path to directory

    Returns:
        target (int): class label

    """
    num_obj = self._get_num_obj(path)
    if num_obj == 0:
      return 1  # pure bg has label 1
    return 0  # remaining images have label 0

  def __getitem__(self, idx):
    filename = self.filenames[idx]
    label = self.labels[idx]
    img = Image.open(filename)
    img = self.transform(img)
    return img, label


# utility functions
def make_equal_label(filenames, labels):
  """
  Duplicate filenames and labels s.t. they have equal numbers of labels.
  Args:
    filenames (list): filenames to duplicate
    labels (list): corresponding label to each filename

  """
  filenames_array = np.array(filenames)
  labels_array = np.array(labels)
  
  counter = Counter(labels)
  max_cnt = max(counter.values())
  
  for lbl, cnt in counter.items():
    if cnt == max_cnt: continue
    diff = max_cnt - cnt
    idcs = np.where(labels_array == lbl)[0]
    
    replace = diff > len(idcs)  # only draw with replacement if necessary
    idcs = np.random.choice(idcs, diff, replace=replace)
    
    filenames.extend(filenames_array[idcs])
    labels.extend(labels_array[idcs])
