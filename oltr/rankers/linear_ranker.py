from oltr.rankers import BaseRanker
import numpy as np


class LinRanker(BaseRanker):
  def __init__(self, weights=None, num_features=136):
    if not weights:
      self.weights = np.mat(np.ones([num_features, 1]))
    else:
      self.weights = np.mat(weights).flatten().T

  def predict(self, X):
    """Get the score of each item.

    Args:
      X: A 2d array of size [num_items, num_features] encoding the features of
        each item.

    Returns:
      A vector of length num_items.
    """
    scores = (np.mat(X) * self.weights).A.flatten()
    return scores
