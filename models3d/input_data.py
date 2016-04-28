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

  train_dirdata = os.path.join(train_dir, "DB-AD-CTRL.pkl")
  test_dirdata= os.path.join(train_dir, "DB-MCIc-MCInc.pkl")

  with open(train_dirdata, "rb") as train_fp, open(test_dirdata, "rb") as test_fp:
      train_data = cp.load(train_fp)
      train_images = np.array(train_data["xs"],dtype=dtype)
      train_labels = np.array( map( lambda x : 0 if x == -1 else 1 , train_data["ys"]),dtype=dtype)

      test_data = cp.load(test_fp)
      test_images = np.array(test_data["xs"],dtype=dtype)
      test_labes = np.array( map( lambda x : 0 if x == -1 else 1 , test_data["ys"]),dtype=dtype)

      #testSplit = 1.0
      #splitSize = int(test_images.shape[0] * testSplit)

      data_sets.train = DataSet(train_images, train_labels, dtype=dtype)
      data_sets.test = DataSet(test_images, test_labes, dtype=dtype)
      data_sets.validation = DataSet(test_images, test_labes, dtype=dtype)

      return data_sets
