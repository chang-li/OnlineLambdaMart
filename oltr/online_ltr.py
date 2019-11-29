import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import timeit
from collections import defaultdict

from oltr.utils.metric import ndcg_at_k
from oltr.utils.click_simulator import DependentClickModel
from oltr.utils.dataset import Data
from oltr.rankers import LinRanker, LMARTRanker, ClickLMARTRanker
from oltr.learners import OnlineLTR, ExploreThenExploitOLTR

DATA_PATH = os.path.expanduser('~/data/web30k/')

NUM_QUERIES_FOR_CLICK_LMART = 10 ** 4

def oltr_loop(data_path, fold=-1, num_iterations=20, num_train_queries=5, num_test_queries=100):
  oltr_ranker_params = {
    'min_child_samples': 50,
    'min_child_weight': 0,
    'n_estimators': 500,
    'learning_rate': 0.02,
    'num_leaves': 400,
    'boosting_type': 'gbdt',
    'objective': 'lambdarank',
  }
  oltr_fit_params = {
    'early_stopping_rounds': 50,
    'eval_metric': 'ndcg',
    'eval_at': 5,
    'verbose': 100,
  }
  eval_params = {
    'metric': ndcg_at_k,
    'cutoff': 10
  }


  fold = '' if fold == -1 else 'Fold%d/' % fold

  data = Data(train_path=data_path+fold+'train.txt',
    valid_path=data_path+fold+'vali.txt',
    test_path=data_path+fold+'test.txt')

  lmart_ranker_params = {
    'min_child_samples': 50,
    'min_child_weight': 0,
    'n_estimators': 500,
    'learning_rate': 0.02,
    'num_leaves': 400,
    'boosting_type': 'gbdt',
    'objective': 'lambdarank',
  }
  lmart_fit_params = {
    'early_stopping_rounds': 50,
    'eval_metric': 'ndcg',
    'eval_at': 5,
    'verbose': 50,
  }
  click_model = DependentClickModel(user_type='pure_cascade')

  # Online learners
  online_learners = {
    # Follow the Leader
    'FTL': OnlineLTR(data.train_qset, data.valid_qset, data.test_qset),
  }
  for num_explore in range(num_iterations):
    online_learners['EtE %d' % num_explore] = ExploreThenExploitOLTR(
      data.train_qset, num_explore, data.valid_qset, data.test_qset)

  online_rankers = {lname: None for lname in online_learners}

  offline_rankers = {
    'Linear': LinRanker(num_features=136),
    'Offline LambdaMART': LMARTRanker(
      data.train_qset, data.valid_qset, data.test_qset,
      lmart_ranker_params, lmart_fit_params),
    'Click LambdaMART': ClickLMARTRanker(
      data.train_qset, data.valid_qset, data.test_qset,
      lmart_ranker_params, lmart_fit_params, click_model=click_model,
      total_number_of_clicked_queries=num_iterations * num_train_queries),
    'Click LambdaMART Random': ClickLMARTRanker(
      data.train_qset, data.valid_qset, data.test_qset,
      lmart_ranker_params, lmart_fit_params, click_model=click_model,
      total_number_of_clicked_queries=num_iterations * num_train_queries, learn_from_random=True),
  }
  eval_results = defaultdict(list)

  for ind in range(num_iterations):
    # Train OLTR
    for lname in online_learners:
      online_rankers[lname] = online_learners[lname].update_learner(
        online_rankers[lname], num_train_queries, click_model,
        oltr_ranker_params, oltr_fit_params)

    # Evaluation
    test_query_ids = online_learners['FTL'].sample_query_ids(num_test_queries,
                                                             data='test')
    # Online
    for lname in online_learners:
      oltr_eval_value = online_learners[lname].evaluate_ranker(
        online_rankers[lname], eval_params, query_ids=test_query_ids)
      eval_results[lname].append(oltr_eval_value)
    # Offline (baselines)
    for offline_model_name, ranker in offline_rankers.items():
      eval_result = online_learners['FTL'].evaluate_ranker(ranker, eval_params,
                                                           query_ids=test_query_ids)
      eval_results[offline_model_name].append(eval_result)

    print('>>>>>>>>>>iteration: ', ind)
    print('Offline LambdaMART (headroom) performance : ',
          eval_results['Offline LambdaMART'][-1])
    # print('Online LTR performance: ', eval_results['OLTR'][-1])
    print('Linear ranker (baseline) performance: ', eval_results['Linear'][-1])
  return eval_results


def plot_eval_results(eval_results, out_path='/tmp/plot.png'):
  fig, ax = plt.subplots()
  for ranker, metrics in eval_results.items():
    ax.plot(metrics, label=ranker)
  print('Saving a plot of the results to', out_path)
  plt.legend(loc='upper left')
  fig.savefig(out_path)


if __name__ == '__main__':
  num_iterations = 10
  num_train_queries = 5
  num_test_queries = 100
  oltr_data_path = '../data/sample/'
  if len(sys.argv) > 1:
    oltr_data_path = DATA_PATH
    num_iterations = int(sys.argv[1])
  if len(sys.argv) > 2:
    num_train_queries = int(sys.argv[2])
  if len(sys.argv) > 3:
    num_test_queries = int(sys.argv[3])
  start = timeit.default_timer()
  eval_results = oltr_loop(oltr_data_path, -1, num_iterations, num_train_queries, num_test_queries)
  plot_eval_results(eval_results,
    out_path='/tmp/oltr_performance_%s_%s_%s.png'
    % (num_iterations, num_train_queries, num_test_queries))
  print('running time: ', timeit.default_timer() - start)
