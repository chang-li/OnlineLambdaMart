import numpy as np
import re
import scipy.sparse as sp


# TODO(mzoghi): Remove these and import the click model
Rel_To_Prob = {
    "perfect": {'c_prob': np.asarray([.0, .2, .4, .8, 1.]),
                's_prob': np.ones(5)},
    "informational": {'c_prob': np.asarray([.4, .6, .7, .8, .9]),
                      's_prob': np.asarray([.1, .2, .3, .4, .5])},
    "navigational": {'c_prob': np.asarray([.05, .3, .5, .7, .95]),
                     's_prob': np.asarray([.2, .3, .5, .7, .9])}
}


class AbstractClickSimulator(object):
    """
    Based class for all simulator
    """
    def __init__(self, user_type):
        self.name = 'abstract Model'

    def get_click(self, r):
        raise NotImplemented

    def __str__(self):
        return self.name


class CascadeModel(AbstractClickSimulator):
    """
    CascadeModel
    """
    def __init__(self, user_type='perfect'):
        super(CascadeModel, self).__init__(user_type)
        self.name = user_type
        self.c_prob = Rel_To_Prob[user_type]['c_prob']
        self.s_prob = Rel_To_Prob[user_type]['s_prob']

    def get_click(self, r):
        pos = len(r)
        c_prob = self.c_prob[r]
        s_prob = self.s_prob[r]
        c_coin = np.random.rand(pos) < c_prob
        s_coin = np.random.rand(pos) < s_prob
        f_coin = np.multiply(c_coin, s_coin)
        if np.sum(f_coin) == 0:
            last_pos = pos
        else:
            last_pos = np.where(f_coin)[0][0]
        c_coin[last_pos+1:] = False
        return c_coin


class OnlineLTR(object):

    # @staticmethod
    # def parse_libsvm_line(line):
    #     # TODO: Extract label qid and features and store them in a dict
    #     #       OR use a libsvm data importer. @Chang: Can you look into this?

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

        qset = {}
        for q_ind, qid in enumerate(query_ids):
            start_ind = query_indptr[q_ind]
            end_ind = query_indptr[q_ind+1]
            qset[qid] = {'feature': feature_vectors[start_ind: end_ind],
                         'relevance': relevances[start_ind: end_ind]}
        return qset

    def __init__(self, data_path='../data/mslr_fold1_test_sample.txt'):
        self.qset = OnlineLTR.load_from_text(data_path)

    def to_df(self):
        """Convert self.qset to dataframes that could be used to train a LightGbM model.

        Returns:
            A tule of dataframes that could be fed into the sklearn API of LightGBM.
        """

    def extract_labels_and_scores(self, ranker):
        """Apply a ranker to the data and get the scores and labels.

        Args:
            ranker: A LightGBM model.

        Returns:
            A pair of numpy arrays that assign labels and scores to the documents of each query.
        """

    def apply_click_model_to_labels_and_scores(self, click_model, ranker, num_queries):
        """This method samples some queries and generates clicks for them based on a click model"""

    def generate_training_data_from_clicks(self):
        """This method uses the clicks geenrated by apply_click_model_to_labels_and_scores to
        create a training dataset."""

    def update_ranker(self):
        """"This method uses the training data from generate_training_data_from_clicks to
        improve the ranker."""
