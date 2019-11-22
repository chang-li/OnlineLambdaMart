import numpy as np
import lightgbm as gbm


class BaseRanker(object):
  def fit(self):
    raise NotImplementedError

  def predict(self):
    raise NotImplementedError


