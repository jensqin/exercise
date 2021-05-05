import pytest
import hypothesis
from nbastats.schema.playbyplay import play_schema

from nbastats.common.playbyplay import common_play


# @hypothesis.given(play_schema.strategy(size=5))
# def test_common_play(dataframe):
#     result = common_play(dataframe)
#     assert "Duration" in result.columns
