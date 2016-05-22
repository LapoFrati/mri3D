import cPickle as cp
import numpy as np
import os
import sys

class DataSet(object):

  def __init__(self, images, labels, dtype=np.float32):
    """ Construct a DataSet """
    self._images = images
    self._labels = labels
    self._num_examples = images.shape[0]
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set. if data is exhausted return a shuffoled version"""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def read_data_sets(train_dir, dtype=np.float32):
  class DataSets(object):
    pass
  data_sets = DataSets()

  #train_dirdata = os.path.join(train_dir, "DB-AD-CTRL.pkl")
  #test_dirdata = os.path.join(train_dir, "DB-MCIc-MCInc.pkl")
  AD_CTRL_dir_data = os.path.join(train_dir, "DB-AD-CTRL.pkl")
  MCIc_MCInc_dir_data = os.path.join(train_dir, "DB-MCIc-MCInc.pkl")

  with open(AD_CTRL_dir_data, "rb") as AD_CTRL_fp, open(MCIc_MCInc_dir_data, "rb") as MCIc_MCInc_fp:
      AD_CTRL_data = cp.load(AD_CTRL_fp)
      AD_CTRL_images = np.array(AD_CTRL_data["xs"],dtype=dtype)
      AD_CTRL_labels = np.array( map( lambda x : 0 if x == -1 else 1 , AD_CTRL_data["ys"]),dtype=dtype)

      MCIc_MCInc_data = cp.load(MCIc_MCInc_fp)
      MCIc_MCInc_images = np.array(MCIc_MCInc_data["xs"],dtype=dtype)
      MCIc_MCInc_labes = np.array( map( lambda x : 0 if x == -1 else 1 , MCIc_MCInc_data["ys"]),dtype=dtype)

      splitfactor = 0.7
      size = len(MCIc_MCInc_images)
      splitSize = size * splitfactor

      train_images = np.concatenate((AD_CTRL_images, MCIc_MCInc_images[0:splitSize]))
      train_labels = np.concatenate((AD_CTRL_labels, MCIc_MCInc_labes[0:splitSize]))

      test_images = MCIc_MCInc_images[splitSize:]
      test_labes = MCIc_MCInc_labes[splitSize:]

      valid_images = test_images
      valid_labes = test_labes

      data_sets.train = DataSet(train_images, train_labels, dtype=dtype)
      data_sets.test = DataSet(test_images, test_labes, dtype=dtype)
      data_sets.validation = DataSet(valid_images, valid_labes, dtype=dtype)

      return data_sets
