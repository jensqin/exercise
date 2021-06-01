import pytest
import hypothesis
from nbastats.schema.example import InputSchema

def processing_fn(df):
    return df.assign(column4=0)

@hypothesis.given(InputSchema.strategy(size=5))
def test_processing_fn(dataframe):
    result = processing_fn(dataframe)
    assert "column4" in result.columns
