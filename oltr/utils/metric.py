import numpy as np


def dcg(y, pos, measure=1):
    """
    :param y:
    :param pos:
    :param measure: 1: 2^rel - 1; 2: rel
    :return: dcg at pos
    """
    pos_ = min(pos, len(y))
    dcg_weights = np.asarray([np.log(2)/np.log(idx+2) for idx in range(pos_)])
    y_ = np.power(2, y) - 1 if measure == 1 else y
    return np.dot(dcg_weights, y_[:pos_])


def ndcg_at_k(y, k, group=None, measure=1):
    """
    Compute ndcg at k
    :param y: labels
    :param k: cut off
    :param group: None: only have a ranking; other have multiple rankings
    :param measure: 1: 2^rel - 1; 2: rel
    :return:
    """
    if not group:
        y_ideal = np.sort(y)[::-1]
        return dcg(y, k)/dcg(y_ideal, k, measure=measure)
    else:
        inds = np.concatenate([[0], group])
        dcgs = [dcg(y[inds[i]: inds[i+1]], k) / dcg(np.sort(y[inds: inds+1])[::-1], k, measure=measure)
                for i in range(len(group))]
        return np.mean(dcgs)

