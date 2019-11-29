import numpy as np

def evaluate_ranker(qset, ranker, eval_params, query_ids=None):
    """ Evaluate the ranker based on the queries in self.train_qset
    :param ranker:
    :param eval_params:  ndcg, cutoff
    :return:
    """
    if query_ids is None:
        eval_qset = qset
    else:
        eval_qset = qset[query_ids]
    scores = ranker.predict(eval_qset.feature_vectors)
    tie_breakers = np.random.rand(scores.shape[0])

    indices = eval_qset.query_indptr
    rankings = [np.lexsort((tie_breakers[indices[i]:indices[i + 1]],
                -scores[indices[i]:indices[i + 1]]))
                for i in range(eval_qset.n_queries)]
    # raise ValueError
    ndcgs = [eval_params['metric'](
                eval_qset[qid].relevance_scores[rankings[qid]],
                eval_params['cutoff'])
             for qid in range(eval_qset.n_queries)]
    return np.mean(ndcgs)
