import numpy as np
import scipy.sparse as sp
import lightgbm as gbm

from oltr.utils.metric import ndcg_at_k
from oltr.utils.click_simulator import DependentClickModel
from oltr.utils.queries import Queries, find_constant_features


class OnlineLTR(object):

    def __init__(self, data_path='../data/mslr_fold1_test_sample.txt', seed=42):

        self.seed = seed
        np.random.seed(seed)

        self.qset = Queries.load_from_text(data_path)
        cls = find_constant_features(self.qset)
        self.qset.adjust(remove_features=cls, purge=True, scale=True)

        # The previous collected training date.
        self.observed_training_data = []

    def get_labels_and_rankings(self, ranker, num_queries):
        """Apply a ranker to a subsample of the data and get the labels and rankings.

        Args:
            ranker: A LightGBM model.
            num_queries: Number of queries to be sampled from self.qset

        Returns:
            A pair of lists that assign labels and rankings to the documents of each query, and the indices of each query.
        """
        query_ids = np.random.choice(self.qset.n_queries, num_queries)
        n_docs_per_query = [self.qset[qid].document_count() for qid in query_ids]
        indices = [0] + np.cumsum(n_docs_per_query).tolist()
        # MZ: @Chang, I don't understand what these indices are or why we need them.
        labels = [self.qset[qid].relevance_scores for qid in query_ids]

        # Get the rankings of document per query
        if ranker is None:
            rankings = [np.random.permutation(n_docs) for n_docs in n_docs_per_query]
        else:
            features = self.qset[query_ids].feature_vectors
            # Cf. https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html#lightgbm.LGBMRanker.predict
            scores = ranker.predict(features)
            tie_breakers = np.random.rand(scores.shape[0])
            rankings = [np.lexsort((tie_breakers[indices[i]:indices[i+1]], -scores[indices[i]:indices[i+1]]))
                        for i in range(num_queries)]

        return query_ids, indices, labels, rankings

    def apply_click_model_to_labels_and_scores(self, click_model, labels, rankings):
        """This method samples some queries and generates clicks for them based on a click model

        Args:
            click_model: a click model
            labels: true labels of documents
            rankings: ranking of documents of each query

        Returns:
            A list of clicks of documents of each query
        """
        clicks = [click_model.get_click(labels[i][rankings[i]]) for i in range(len(rankings))]
        return clicks

    def generate_training_data_from_clicks(self, query_ids, clicks, rankings):
        """This method uses the clicks geenrated by apply_click_model_to_labels_and_scores to
        create a training dataset.

        Args:
            query_ids: the sampled query ids
            indices: indices of each query
            clicks: clicks from the click model

        Returns:
            A tuple of (train_features, train_labels):
                train_features: list of observed docs per query
                train_labels: list of click feedback per query
        """
        # last observed position of each ranking
        last_pos = []
        train_labels = []
        for click in clicks:
            if sum(click) == 0:
                last_pos.append(len(click))
            else:
                last_pos.append(np.where(click)[0][-1]+1)
            train_labels.append(click[:last_pos[-1]])

        train_features = [self.qset[query_ids[i]].feature_vectors[rankings[i]][:last_pos[i]]
                          for i in range(len(query_ids))]

        # Cf. the following for an example:
        # https://mlexplained.com/2019/05/27/learning-to-rank-explained-with-code/
        train_qid_list = [feature.shape[0] for feature in train_features]

        train_features = np.concatenate(train_features)
        train_labels = np.concatenate(train_labels)

        self.observed_training_data.append((train_features, train_labels, train_qid_list))
        return (train_features, train_labels, train_qid_list)

    def update_ranker(self, training_data, ranker_params, fit_params):
        """"This method uses the training data from generate_training_data_from_clicks to
        improve the ranker."""
        if self.observed_training_data:
            train_features = np.concatenate([otd[0] for otd in self.observed_training_data])
            train_labels = np.concatenate([otd[1] for otd in self.observed_training_data])
            train_qid_list = np.concatenate([otd[2] for otd in self.observed_training_data])
        else:
            train_features, train_labels, train_qid_list = training_data

        ranker = gbm.LGBMRanker(**ranker_params)
        ranker.fit(X=train_features, y=train_labels, group=train_qid_list)
        return ranker

    def evalualte_ranker(self, ranker, eval_params):
        """ Evaluate the ranker based on the queries in self.qset
        :param ranker:
        :param eval_params:  ndcg, cutoff
        :return:
        """
        scores = ranker.predict(self.qset.feature_vectors)
        tie_breakers = np.random.rand(scores.shape[0])

        indices = self.qset.query_indptr
        rankings = [np.lexsort((tie_breakers[indices[i]:indices[i + 1]], -scores[indices[i]:indices[i + 1]]))
                    for i in range(self.qset.n_queries)]
        ndcgs = [eval_params['metric'](self.qset[qid].relevance_scores[rankings[qid]], eval_params['cutoff'])
                 for qid in range(self.qset.n_queries)]
        return np.mean(ndcgs)


def oltr_loop(data_path, num_iterations=10, num_queries=5):
    learner = OnlineLTR(data_path)
    ranker_params = {
        'min_child_samples': 50,
        'min_child_weight': 0,
        'n_estimators': 500,
        'learning_rate': 0.02,
        'num_leaves': 400,
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',
    }
    fit_params = {
        'early_stopping_rounds': 50,
        'eval_metric': 'ndcg',
        'eval_at': 5,
        'verbose': 5,
    }
    eval_params = {
        'metric': ndcg_at_k,
        'cutoff': 10
    }
    click_model = DependentClickModel(user_type='pure_cascade')

    ranker = None
    for ind in range(num_iterations):
        query_ids, indices, labels, rankings = learner.get_labels_and_rankings(ranker, num_queries)
        clicks = learner.apply_click_model_to_labels_and_scores(click_model, labels, rankings)
        training_data = learner.generate_training_data_from_clicks(query_ids, clicks, rankings)
        ranker = learner.update_ranker(training_data, ranker_params, fit_params)
        eval_value = learner.evalualte_ranker(ranker, eval_params)

        print('>>>>>>>>>>')
        print('iteration: ', ind)
        print('number of clicks: ', sum([sum(ck) for ck in clicks]))
        print('evaluation value: ', eval_value)


if __name__ == '__main__':
    oltr_loop('../data/mslr_fold1_test_sample.txt')
