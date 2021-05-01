import pytest
import hypothesis
from nbastats.schema.example import InputSchema

def processing_fn(df):
    return df.assign(column4=df.column1 * df.column2)

@hypothesis.given(InputSchema.strategy(size=5))
def test_processing_fn(dataframe):
    result = processing_fn(dataframe)
    assert "column4" in result