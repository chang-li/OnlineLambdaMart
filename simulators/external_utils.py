# This file is part of RankPy.
#
# RankPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RankPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RankPy.  If not, see <http://www.gnu.org/licenses/>.


import os
import numpy as np
import numbers
import scipy.sparse

try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle


def pickle(obj, filepath, protocol=-1):
    '''
    Pickle the object into the specified file.

    Parameters:
    -----------
    obj: object
        The object that should be serialized.

    filepath:
        The location of the resulting pickle file.
    '''
    with open(filepath, 'wb') as fout:
        _pickle.dump(obj, fout, protocol=protocol)


def unpickle(filepath):
    '''
    Unpicle the object serialized in the specified file.

    Parameters:
    -----------
    filepath:
        The location of the file to unpickle.
    '''
    with open(filepath) as fin:
        return _pickle.load(fin)

def asindexarray(x):
    '''
    Helper method which converts the given parameter into a
    list of indices.

    Returns
    -------
    indices : array of ints
        The index array created from `x`.
    '''
    if isinstance(x, (numbers.Integral, np.integer)):
        return np.array([x], dtype='int32', order='C')

    if isinstance(x, slice):
        return np.arange(x.start, x.stop, x.step, dtype='int32')

    if isinstance(x, np.ndarray):
        if x.ndim != 1:
            raise ValueError('index array has more than 1 dimension')

        if x.dtype.kind == 'b':
            return x.nonzero()[0].astype(dtype='int32', order='C')

        return x.astype('int32', order='C', casting='same_kind')

    # Assuming x is a list.
    if isinstance(x[0], (numbers.Integral, np.integer)):
        return np.array(x, dtype='int32', order='C')
    else:
        raise ValueError('input cannot be converted to an index array')