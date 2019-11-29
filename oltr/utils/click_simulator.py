import numpy as np

Rel_To_Prob = {
    "perfect": {'c_prob': np.asarray([.0, .2, .4, .8, 1.]),
                's_prob': np.zeros(5)},
    "informational": {'c_prob': np.asarray([.4, .6, .7, .8, .9]),
                      's_prob': np.asarray([.1, .2, .3, .4, .5])},
    "navigational": {'c_prob': np.asarray([.05, .3, .5, .7, .95]),
                     's_prob': np.asarray([.2, .3, .5, .7, .9])},
    "pure_cascade": {'c_prob': np.asarray([.05, .3, .5, .7, .95]),
                     's_prob': np.ones(5)},
}


class AbstractClickSimulator(object):
    """
    Based class for all simulator
    """
    def __init__(self, user_type):
        self.name = 'abstract Model'

    def get_click(self, r):
        raise NotImplementedError

    def __str__(self):
        return self.name


class DependentClickModel(AbstractClickSimulator):
    """
    DependentClickModel
    """
    def __init__(self, user_type='perfect'):
        super(DependentClickModel, self).__init__(user_type)
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


class PBM(AbstractClickSimulator):
    def __init__(self, user_type, e_prob=None):
        super(PBM, self).__init__(user_type)
        self.name = 'PBM' + user_type
        self.c_prob = Rel_To_Prob[user_type]['c_prob']
        if not e_prob:
            self.e_prob = np.asarray([0.99997132, 0.95949374, 0.76096783, 0.59179909, 0.45740329, 0.38584302, 0.33052186,
                                      0.28372475, 0.26700303, 0.26211924])
        else:
            self.e_prob = e_prob

    def get_click(self, r):
        assert len(r) <= len(self.e_prob)
        c_prob = np.multiply(self.c_prob[r], self.e_prob[:len(r)])
        return np.random.rand(len(r)) < c_prob

