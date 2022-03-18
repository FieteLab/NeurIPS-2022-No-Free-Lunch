import numpy as np
from itertools import product

def dict_to_str(adict):
    '''Converts a dictionary (e.g. hyperparameter configuration) into a string'''
    return ''.join('{}{}'.format(key, val) for key, val in sorted(adict.items()))

def iterate_dicts(inp):
    '''Computes cartesian product of parameters
    From: https://stackoverflow.com/questions/10472907/how-to-convert-dictionary-into-string'''
    return list((dict(zip(inp.keys(), values)) for values in product(*inp.values())))

def check_np_equal(a, b):
    '''Checks two numpy arrays are equal and works with nan values unlike np.array_equal.
    From: https://stackoverflow.com/questions/10710328/comparing-numpy-arrays-containing-nan'''
    return ((a == b) | (np.isnan(a) & np.isnan(b))).all()

def get_params_from_workernum(worker_num, param_lookup):
    return param_lookup[worker_num]

def all_disjoint(sets):
    '''Checks if a list of sets is all pairwise disjoint.
    From: https://stackoverflow.com/questions/22432814/check-if-a-collection-of-sets-is-pairwise-disjoint'''
    union = set()
    for s in sets:
        for x in s:
            if x in union:
                return False
            union.add(x)
    return True

def get_shape_2d(num_states=None, shape_2d=None):
    if shape_2d is None:
        num_rows = (int)(np.sqrt(num_states))
        # has to be perfect square if shape_2d is not specified
        assert(np.array_equal(np.sqrt(num_states), (float)(num_rows)))
        num_cols = num_rows
    else:
        assert(len(shape_2d) == 2)
        num_rows = shape_2d[0]
        num_cols = shape_2d[1]

    return num_rows, num_cols

def get_kw_from_map(map_type):
    if map_type == "ridge":
        map_kwargs = {"map_type": "percentile", "map_kwargs": {"percentile": 0, "identity": False}}
    elif map_type == "1-1":
        map_kwargs = {"map_type": "percentile", "map_kwargs": {"percentile": 100, "identity": True}}
    elif map_type == "pls100":
        map_kwargs = {"map_type": "pls", "map_kwargs": {"n_components": 100}}
    else:
        raise ValueError

    return map_kwargs