import numpy as np
import scipy.sparse as sp
import lightgbm as gbm

from simulators.click_simulator import DependentClickModel
# Todo (Chang): re-name the folders.


class OnlineLTR(object):

    @staticmethod
    def load_from_text(data_path, dtype=np.float32, min_feature=None, max_feature=None, purge=False):
        """ Load from the file and return the query dict.
        :return: qset.
        :param min_feature: minimal number of features
        :param max_feature: maximal number of features
        :param purge: bool, if True remove query that contain documents with the same relevance
        """

        # Arrays used to build CSR matrix of query-document vectors.
        data, indices, indptr = [], [], [0]

        # Relevance score, query ID, query hash, and document hash.
        relevances = []
        query_ids = []
        query_indptr = [0]
        prev_qid = None

        # If only single filepath is given, not a list.
        if isinstance(data_path, str):
            data_path = [data_path]

        n_purged_queries = 0
        n_purged_documents = 0

        def purge_query(qid, data, indices, indptr):
            """Remove the last query if all relevances of documents are the same."""
            if not purge or qid is None:
                return 0

            r = relevances[query_indptr[-2]]

            i = query_indptr[-2]
            while i < query_indptr[-1] and relevances[i] == r:
                i += 1

            if i == query_indptr[-1]:
                n = query_indptr.pop()

                del query_ids[-1]

                del indices[indptr[query_indptr[-1]]:]
                del data[indptr[query_indptr[-1]]:]

                del relevances[query_indptr[-1]:]
                del indptr[query_indptr[-1] + 1:]

                return n - query_indptr[-1]
            else:
                return 0

        for filepath in data_path:
            lineno = 0  # Used just to report invalid lines (if any).

            print('Reading queries from %s.' % filepath)

            with open(filepath, 'r') as ifile:
                # Loop through every line containing query-document pair.
                for pair in ifile:
                    lineno += 1
                    try:
                        pair = pair.strip()

                        # Skip comments and empty lines.
                        if not pair:
                            continue

                        items = pair.split()

                        # Query ID follows the second item on the line,
                        # which is 'qid:'.
                        qid = int(items[1].split(':')[1])

                        if qid != prev_qid:
                            # Make sure query is sanitized before being
                            # added to the set.
                            n_purged = purge_query(prev_qid, data,
                                                   indices, indptr)

                            n_purged_documents += n_purged

                            if n_purged > 0:
                                print('Ignoring query %d (qid) with %d '
                                             'documents because all had the '
                                             'same relevance label.'
                                             % (prev_qid, n_purged))
                                n_purged_queries += 1

                            query_ids.append(qid)
                            query_indptr.append(query_indptr[-1] + 1)
                            prev_qid = qid

                        else:
                            query_indptr[-1] += 1

                        # Relevance is the first number on the line.
                        relevances.append(int(items[0]))

                        # Load the feature vector into CSR arrays.
                        for fidx, fval in map(lambda s: s.split(':'),
                                              items[2:]):
                            data.append(dtype(fval))
                            indices.append(int(fidx))

                        indptr.append(len(indices))

                        if (query_indptr[-1] + n_purged_documents) % 10000 == 0:
                            print('Read %d queries and %d documents '
                                        'so far.' % (len(query_indptr) +
                                                     n_purged_queries - 1,
                                                     query_indptr[-1] +
                                                     n_purged_documents))
                    except:
                        # Ill-formated line (it should not happen).
                        # Print line number
                        print('Ill-formated line: %d' % lineno)
                        raise

                # Need to check the last added query.
                n_purged = purge_query(prev_qid, data, indices, indptr)
                n_purged_documents += n_purged

                if n_purged > 0:
                    print('Ignoring query %d (qid) with %d documents '
                                 'because all had the same relevance label.'
                                 % (prev_qid, n_purged))
                    n_purged_queries += 1

                print('Read %d queries and %d documents out of which '
                            '%d queries and %d documents were discarded.'
                            % (len(query_indptr) + n_purged_queries - 1,
                               query_indptr[-1] + n_purged_documents,
                               n_purged_queries, n_purged_documents))

        # Set the minimum feature ID, if not given.
        if min_feature is None:
            min_feature = min(indices)
        if max_feature is None:
            # Remap the features for a proper conversion into dense matrix.
            feature_indices = np.unique(np.r_[min_feature, indices])
            indices = np.searchsorted(feature_indices, indices)
        else:
            feature_indices = np.arange(min_feature,
                                        max_feature + 1,
                                        dtype='int32')

            indices = np.array(indices, dtype='int32') - min_feature

        feature_vectors = sp.csr_matrix((data, indices, indptr), dtype=dtype,
                                        shape=(query_indptr[-1],
                                               len(feature_indices)))
        del data, indices, indptr

        relevances = np.asarray(relevances)
        qset = {}
        for q_ind, qid in enumerate(query_ids):
            start_ind = query_indptr[q_ind]
            end_ind = query_indptr[q_ind+1]
            qset[qid] = {'feature': feature_vectors[start_ind: end_ind].toarray(),
                         'label': relevances[start_ind: end_ind]}
        return qset

    def __init__(self, data_path='../data/mslr_fold1_test_sample.txt', seed=42):
        self.seed = seed
        np.random.seed(seed)

        self.qset = OnlineLTR.load_from_text(data_path)

    def get_labels_and_rankings(self, ranker, num_queries):
        """Apply a ranker to a subsample of the data and get the labels and rankings.

        Args:
            ranker: A LightGBM model.
            num_queries: Number of queries to be sampled from self.qset

        Returns:
            A pair of lists that assign labels and rankings to the documents of each query, and the indices of each query.
        """
        query_ids = np.random.choice(list(self.qset), num_queries)
        n_docs_per_query = [self.qset[qid]['label'].shape[0] for qid in query_ids]
        indices = [0] + np.cumsum(n_docs_per_query).tolist()
        # MZ: @Chang, I don't understand what these indices are or why we need them.
        labels = [self.qset[qid]['label'] for qid in query_ids]

        total_docs = sum([len(label) for label in labels])
        # Get the labels from self.qset
        if ranker is None:
            rankings = [np.random.permutation(label.shape[0]) for label in labels]
        else:
            features = np.concatenate([self.qset[qid]['feature'] for qid in query_ids])
            # Cf. https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html#lightgbm.LGBMRanker.predict
            scores = ranker.predict(features)
            tie_breakers = np.random.rand(scores.shape[0])
            rankings = [np.lexsort((tie_breakers[indices[i]:indices[i+1]], -scores[indices[i]:indices[i+1]]))for i in range(num_queries)]

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

        train_features = [self.qset[query_ids[i]]['feature'][rankings[i]][:last_pos[i]]
                          for i in range(len(query_ids))]


        # Cf. the following for an example:
        # https://mlexplained.com/2019/05/27/learning-to-rank-explained-with-code/
        train_qid_list = [feature.shape[0] for feature in train_features]

        train_features = np.concatenate(train_features)
        train_labels = np.concatenate(train_labels)
        return (train_features, train_labels, train_qid_list)

    def update_ranker(self, training_data, ranker_params, fit_params):
        """"This method uses the training data from generate_training_data_from_clicks to
        improve the ranker."""
        train_features, train_labels, train_qid_list = training_data
        ranker = gbm.LGBMRanker(**ranker_params)
        ranker.fit(X=train_features, y=train_labels, group=train_qid_list)
        return ranker




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
    click_model = DependentClickModel(user_type='pure_cascade')

    ranker = None
    for ind in range(num_iterations):
        query_ids, indices, labels, rankings = learner.get_labels_and_rankings(ranker, num_queries)
        clicks = learner.apply_click_model_to_labels_and_scores(click_model, labels, rankings)
        training_data = learner.generate_training_data_from_clicks(query_ids, clicks, rankings)
        ranker = learner.update_ranker(training_data, ranker_params, fit_params)

        print('iteration: ', ind)
        print('number of clicks', sum([sum(ck) for ck in clicks]))
        print('number of training samples', sum(td.shape[0] for td in training_data[0]))


if __name__ == '__main__':
    oltr_loop('../data/mslr_fold1_test_sample.txt')
