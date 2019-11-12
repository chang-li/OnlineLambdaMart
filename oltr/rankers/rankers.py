import numpy as np
import lightgbm as gbm

from oltr.utils.queries import Queries, find_constant_features


class BaseRanker(object):
  def fit(self):
    raise NotImplementedError

  def predict(self):
    raise NotImplementedError


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


class LMARTRanker(BaseRanker):

  def __init__(self, train_path, valid_path, test_path,
               ranker_params, fit_params):
    self.train_qset = Queries.load_from_text(train_path)
    self.valid_qset = Queries.load_from_text(valid_path)
    self.test_qset = Queries.load_from_text(test_path)
    self.ranker_params = ranker_params
    self.fit_params = fit_params
    self.ranker = gbm.LGBMRanker(**self.ranker_params)
    self.fit()

  def train_and_eval(self):
    ranker = gbm.LGBMRanker(**self.ranker_params)
    train_features = self.train_qset.feature_vectors
    valid_features = self.valid_qset.feature_vectors
    test_features = self.test_qset.feature_vectors
    train_labels = self.train_qset.relevance_scores
    train_qid = np.diff(self.train_qset.query_indptr)
    ranker.fit(X=train_features, y=train_labels, group=train_qid, **self.fit_params)

  def fit(self):
    """
      Train the ranker on the training set
    """
    train_features = self.train_qset.feature_vectors
    train_labels = self.train_qset.relevance_scores
    train_qid = np.diff(self.train_qset.query_indptr)
    valid_features = self.valid_qset.feature_vectors
    valid_labels = self.valid_qset.relevance_scores
    valid_qid = np.diff(self.valid_qset.query_indptr)
    self.ranker.fit(X=train_features, y=train_labels, group=train_qid,
                    eval_set=[(valid_features, valid_labels)], eval_group=[valid_qid],
                    **self.fit_params)

  def predict(self, X):
    """Get the score of each item.

    Args:
      X: A 2d array of size [num_items, num_features] encoding the features of
        each item.

    Returns:
      A vector of length num_items.
    """
    return self.ranker.predict(X)
