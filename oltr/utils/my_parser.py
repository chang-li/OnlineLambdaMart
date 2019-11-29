import numpy as np
from argparse import ArgumentParser
import logging


class SimulationArgumentParser(ArgumentParser):
    """
    Arguments used for the online simulation experiments.
    """

    def __init__(self, description=None, set_arguments={}):
        self.description = description
        self._initial_set_arguments = dict(set_arguments)
        self.set_arguments = set_arguments
        self.argument_namespaces = {}
        super(SimulationArgumentParser, self).__init__(description=description)

        self.add_argument('--data', '--data_name', dest='data_name', default='mslr', type=str, help='dataset name')
        self.add_argument('-C', '--clickmodel', dest='ClickModel', default='pure_cascade', type=str, help='click model')
        self.add_argument('-R', '--repeat', dest='repeat', default=1, type=int, help='number of repeats')
        self.add_argument('-K', '--n_pos', dest='K', default=5, type=int, help='number of positions')
        self.add_argument('--num_train_queries', dest='num_train_queries', default=5, type=int, help='number of training queries')
        self.add_argument('--num_test_queries', dest='num_test_queries', default=-1, type=int, help='number of testing queries')
        self.add_argument('--n_jobs', dest='n_jobs', default=1, type=int, help='number of jobs')
        self.add_argument('--iter', '--iteration', dest='num_iterations', default=int(10), type=int, help='iterations')
        self.add_argument('--seed', dest='seed', default=42, type=int, help='random seed')
        self.add_argument('-O', '--output', dest='output', default='../results/', type=str, help='output directory')

    @staticmethod
    def print(args):
        arg_dict = vars(args)
        for key in arg_dict:
            print(key, arg_dict[key])


if __name__ == '__main__':
    sim_args = SimulationArgumentParser()
    args = sim_args.parse_args()
    arg_dict = vars(args)
    for key in arg_dict:
        print(key, arg_dict[key])