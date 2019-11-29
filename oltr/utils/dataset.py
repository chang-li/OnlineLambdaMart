from oltr.utils.queries import Queries, find_constant_features 


DATA_PATH = {
    'debug': {
        'train_path': '../../../data/mslr_fold1_train_sample.txt', 
        'test_path': '../../../data/mslr_fold1_test_sample.txt',
        'valid_path': '../../../data/mslr_fold1_valid_sample.txt'
    }
}


class Data(object):
  def __init__(self, train_path, valid_path, test_path):
    if train_path.endswith('.txt'):
      try:
        print('Data: Loading data from', train_path[:-4])
        self.train_qset = Queries.load(train_path[:-4])
      except FileNotFoundError:
        print('Data: Loading data from', train_path)
        self.train_qset = Queries.load_from_text(train_path, purge=True)
        self.train_qset.save(train_path[:-4])
    if valid_path.endswith('.txt'):
      try:
        print('Data: Loading data from', valid_path[:-4])
        self.valid_qset = Queries.load(valid_path[:-4])
      except FileNotFoundError:
        print('Data: Loading data from', valid_path)
        self.valid_qset = Queries.load_from_text(valid_path, purge=True)
        self.valid_qset.save(valid_path[:-4])
    if test_path.endswith('.txt'):
      try:
        print('Data: Loading data from', test_path[:-4])
        self.test_qset = Queries.load(test_path[:-4])
      except FileNotFoundError:
        print('Data: Loading data from', test_path)
        self.test_qset = Queries.load_from_text(test_path, purge=True)
        self.test_qset.save(test_path[:-4])
