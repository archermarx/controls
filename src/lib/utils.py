import numpy as np


def clip_to_bounds(x, lb, ub):
    _x, _lb, _ub = np.array(x), np.array(lb), np.array(ub)
    return np.minimum(_ub, np.maximum(_lb, _x))


def in_bounds(x, lb, ub):
    _x, _lb, _ub = np.array(x), np.array(lb), np.array(ub)
    return np.all(_x >= _lb) and np.all(_x <= _ub)


def scale_to_bounds(x, lb, ub):
    _x, _lb, _ub = np.array(x), np.array(lb), np.array(ub)
    width = _ub - _lb
    return (_x - _lb) / width


def unscale_to_bounds(x, lb, ub):
    # The thing being unscaled should be between 0 and 1
    _x, _lb, _ub = np.array(x), np.array(lb), np.array(ub)
    assert np.all(_x >= 0) and np.all(_x <= 1)
    return (_ub - _lb) * _x + _lb
