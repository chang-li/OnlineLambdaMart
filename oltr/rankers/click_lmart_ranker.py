from oltr.rankers import BaseRanker
import numpy as np
import lightgbm as gbm


class ClickLMARTRanker(BaseRanker):

  def __init__(self, train_qset, valid_qset, test_qset,
               ranker_params, fit_params, click_model,
               total_number_of_clicked_queries=10000):
    self.fit_params = fit_params
    self.click_model = click_model
    self.offline_train_qset = train_qset
    self.offline_valid_qset = valid_qset
    self.offline_test_qset = test_qset
    self.offline_qset = {'train': train_qset, 'test': test_qset, 'valid': valid_qset}
    self.ranker_params = ranker_params
    self.fit_params = fit_params
    self.offline_ranker = gbm.LGBMRanker(**self.ranker_params)
    self.offline_fit()
    self.click_ranker = gbm.LGBMRanker(**self.ranker_params)
    self.click_fit(total_number_of_clicked_queries)

  def offline_fit(self):
    """
      Train the ranker on the training set
    """
    train_features = self.offline_train_qset.feature_vectors
    train_labels = self.offline_train_qset.relevance_scores
    train_qid = np.diff(self.offline_train_qset.query_indptr)
    valid_features = self.offline_valid_qset.feature_vectors
    valid_labels = self.offline_valid_qset.relevance_scores
    valid_qid = np.diff(self.offline_valid_qset.query_indptr)
    self.offline_ranker.fit(X=train_features, y=train_labels, group=train_qid,
                    eval_set=[(valid_features, valid_labels)], eval_group=[valid_qid],
                    **self.fit_params)

  def click_fit(self, total_number_of_clicked_queries):
    """
      Train the ranker on the click training set
    """
    self.num_training_queries = {
        'train': int(.6 * total_number_of_clicked_queries),
        'valid': int(.2 * total_number_of_clicked_queries),
        'test': int(.2 * total_number_of_clicked_queries),
        }
    self.click_training_data = {}
    for data in ['train', 'valid', 'test']:
      query_ids, labels, rankings = \
        self.get_labels_and_rankings(self.offline_ranker,
                                     self.num_training_queries[data],
                                     data)
      clicks = self.apply_click_model_to_labels_and_scores(self.click_model,
                                                           labels,
                                                           rankings)
      self.click_training_data[data] = \
        self.generate_training_data_from_clicks(query_ids, clicks,
                                                rankings, data)
    features = {}
    labels = {}
    q_list_sizes = {}
    for data in ['train', 'valid', 'test']:
      indices = self.click_training_data[data][0]
      features[data] = np.concatenate([self.offline_qset[data].feature_vectors[idx] for idx in indices])
      labels[data] = self.click_training_data[data][1]
      q_list_sizes[data] = self.click_training_data[data][2]
    self.click_ranker.fit(X=features['train'], y=labels['train'],
                          group=q_list_sizes['train'],
                          eval_set=[(features['valid'], labels['valid'])],
                          eval_group=[q_list_sizes['valid']],
                          **self.fit_params)

  def predict(self, X):
    """Get the score of each item.

    Args:
      X: A 2d array of size [num_items, num_features] encoding the features of
        each item.

    Returns:
      A vector of length num_items.
    """
    return self.click_ranker.predict(X)

  def sample_query_ids(self, num_queries, data='train'):
    qset = self.offline_train_qset
    if data == 'valid':
      qset = self.offline_valid_qset
    if data == 'test':
      qset = self.offline_test_qset
    return np.random.choice(qset.n_queries, num_queries)

  def get_labels_and_rankings(self, ranker, num_queries, data='train'):
    """Apply a ranker to a subsample of the data and get the labels and ranks.

    Args:
      ranker: A LightGBM model.
      num_queries: Number of queries to be sampled from self.offline_train_qset

    Returns:
      A tuple of lists that assign labels and rankings to the documents of each
      query.
    """
    qset = self.offline_train_qset
    if data == 'valid':
      qset = self.offline_valid_qset
    if data == 'test':
      qset = self.offline_test_qset
    query_ids = self.sample_query_ids(num_queries)
    n_docs_per_query = [qset[qid].document_count() for qid in query_ids]
    indices = [0] + np.cumsum(n_docs_per_query).tolist()
    labels = [qset[qid].relevance_scores for qid in query_ids]

    # Get the rankings of document per query
    if ranker is None:
      rankings = [np.random.permutation(n_docs) for n_docs in n_docs_per_query]
    else:
      features = qset[query_ids].feature_vectors
      scores = ranker.predict(features)
      tie_breakers = np.random.rand(scores.shape[0])
      rankings = [np.lexsort((tie_breakers[indices[i]:indices[i+1]],
                              -scores[indices[i]:indices[i+1]]))
                  for i in range(num_queries)]

    return query_ids, labels, rankings

  def apply_click_model_to_labels_and_scores(self, click_model, labels,
                                             rankings):
    """This method samples some queries and generates clicks for them based on
    a click model.

    Args:
      click_model: a click model
      labels: true labels of documents
      rankings: ranking of documents of each query

    Returns:
      A list of clicks of documents of each query
    """
    clicks = [click_model.get_click(labels[i][rankings[i]])
              for i in range(len(rankings))]
    return clicks

  def generate_training_data_from_clicks(self, query_ids, clicks, rankings,
                                         data='train'):
    """This method uses the clicks generated by
    apply_click_model_to_labels_and_scores to create a training dataset.

    Args:
      query_ids: the sampled query ids
      clicks: clicks from the click model

    Returns:
      A tuple of (features, labels):
        features: list of observed docs per query
        labels: list of click feedback per query
    """
    # last observed position of each ranking
    qset = self.offline_train_qset
    if data == 'valid':
      qset = self.offline_valid_qset
    if data == 'test':
      qset = self.offline_test_qset
    last_pos = []
    labels = []
    for click in clicks:
      if sum(click) == 0:
        last_pos.append(len(click))
      else:
        last_pos.append(np.where(click)[0][-1]+1)
      labels.append(click[:last_pos[-1]])

    indices = [qset.query_indptr[qid] + rankings[i][:last_pos[i]]
                     for i, qid in enumerate(query_ids)]
    features = [qset.feature_vectors[idx] for idx in indices]

    # Cf. the following for an example:
    # https://mlexplained.com/2019/05/27/learning-to-rank-explained-with-code/
    q_list_sizes = [feature.shape[0] for feature in features]

    # features = np.concatenate(features)
    labels = np.concatenate(labels)

    return (indices, labels, q_list_sizes)


