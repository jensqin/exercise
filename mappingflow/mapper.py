import pandas as pd 
import fuzzywuzzy as fuzz 

class BaseMapper:
    """
    Merge Two DataFrames.
    """
    def __init__(self, df_left, df_right, *args, **kwargs):
        self.left = df_left
        self.right = df_right
        self.merged = None
        self.left_mismatch = None
        self.right_mismatch = None

    def __repr__(self):
        leftstr = f"Left DataFrame Has {len(self.left.index)} Rows, "
        rightstr = f"Right DataFrame Has {len(self.right.index)} Rows, "
        mergedstr = f"Merged DataFrame Has {len(self.merged.index)} Rows."
        return (leftstr + rightstr + mergedstr)

    def flow(self):
        raise AssertionError

    def __call__(self):
        a = 1
    
    def ExactMerge(self, *args, **kwargs):
        df_merged = pd.merge(self.left, self.right, **kwargs)

    def FuzzyMerge(self, *args, **kwargs):
        df_merged = pd.merge(self.left, self.right, **kwargs)

class layer:
    """
    Mapping layer.
    """
    def __init__(self):
        self.name = 'what'

class ExactMerge(layer):
    """
    asdfasdf
    """

import functools
class what:

    def __init__(self, x):
        self.v1 = x
        self.v2 = x + 1

    def _deco(func):
        @functools.wraps(func)
        def wrapper(self):
            func(self)
            print('this is deco')
        return wrapper

    _deco = staticmethod(_deco)

    @_deco
    def eg1(self):
        print('this is eg1')

    @_deco 
    def eg2(self):
        print('this is eg2')
