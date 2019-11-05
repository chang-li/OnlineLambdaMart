import numpy as np
import re


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


class LTRData:
	@statismethod
	def parse_libsvm_line(line):
		# TODO: Extract label qid and features and store them in a dict
		#       OR use a libsvm data importer. @Chang: Can you look into this?


	def __init__(self, data_path='../data/mslr_fold1_test_sample.txt'):
		with open(data_path) as fh:
			query_doc_pairs = [LTRData.parse_libsvm_line(line) for line in fh.readlines()]
		self.qset = {}
		for qd_pair in query_doc_pairs:
			self.qset[qd_pair['qid']] = qd_pair

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