from oltr.rankers import BaseRanker
import numpy as np
import lightgbm as gbm


class LMARTRanker(BaseRanker):

  def __init__(self, train_qset, valid_qset, test_qset,
               ranker_params, fit_params):
    self.train_qset = train_qset
    self.valid_qset = valid_qset
    self.test_qset = test_qset
    self.ranker_params = ranker_params
    self.fit_params = fit_params
    self.ranker = gbm.LGBMRanker(**self.ranker_params)
    self.fit()

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

