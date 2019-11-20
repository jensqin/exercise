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
    