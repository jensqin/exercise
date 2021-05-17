import pandera as pa
from pandera import (
    DataFrameSchema,
    Column,
    Check,
    Index,
    MultiIndex,
    PandasDtype,
)

play_schema = DataFrameSchema(
    columns={
        "GameId": Column(pandas_dtype=PandasDtype.Int64),
        "PlayNum": Column(
            pandas_dtype=PandasDtype.Int64,
            checks=Check.greater_than_or_equal_to(min_value=0.0),
        ),
        "Eventmsgtype": Column(
            pandas_dtype=PandasDtype.Int64,
            checks=[
                Check.greater_than_or_equal_to(min_value=1.0),
                Check.less_than_or_equal_to(max_value=13.0),
            ],
        ),
        "Season": Column(
            pandas_dtype=PandasDtype.Int64,
            checks=Check.greater_than_or_equal_to(min_value=1980),
        ),
        "HomeTeamId": Column(pandas_dtype=PandasDtype.Float64, nullable=True),
        "HomePlayer\dId": Column(
            pandas_dtype=PandasDtype.Float64, nullable=True, regex=True,
        ),
        "AwayTeamId": Column(pandas_dtype=PandasDtype.Float64, nullable=True),
        "AwayPlayer\dId": Column(
            pandas_dtype=PandasDtype.Float64, nullable=True, regex=True,
        ),
        "SecRemainGame": Column(
            pandas_dtype=PandasDtype.Int64,
            checks=Check.greater_than_or_equal_to(min_value=0.0),
        ),
        "Period": Column(
            pandas_dtype=PandasDtype.Int64,
            checks=[
                Check.greater_than_or_equal_to(min_value=1.0),
                Check.less_than_or_equal_to(max_value=10.0),
            ],
        ),
        "SecRemainPeriod": Column(
            pandas_dtype=PandasDtype.Int64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                Check.less_than_or_equal_to(max_value=720.0),
            ],
        ),
        "ScoreMargin": Column(
            pandas_dtype=PandasDtype.Int64,
            checks=[
                Check.greater_than_or_equal_to(min_value=-70.0),
                Check.less_than_or_equal_to(max_value=70.0),
            ],
        ),
        "HomeOff": Column(
            pandas_dtype=PandasDtype.Float64, checks=Check.isin([0, 1]), nullable=True,
        ),
        "StartEvent": Column(pandas_dtype=PandasDtype.String, nullable=True),
        "EndEvent": Column(pandas_dtype=PandasDtype.String, nullable=True),
        "HomePlayer\dEvent": Column(
            pandas_dtype=PandasDtype.String, nullable=True, regex=True
        ),
        "AwayPlayer\dEvent": Column(
            pandas_dtype=PandasDtype.String, nullable=True, regex=True
        ),
        "HomeNumPlayers": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=1.0),
                Check.less_than_or_equal_to(max_value=6.0),
            ],
            nullable=True,
        ),
        "AwayNumPlayers": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=1.0),
                Check.less_than_or_equal_to(max_value=6.0),
            ],
            nullable=True,
        ),
        "HomeFouls": Column(
            pandas_dtype=PandasDtype.Int64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                # Check.less_than_or_equal_to(max_value=18.0),
            ],
        ),
        "AwayFouls": Column(
            pandas_dtype=PandasDtype.Int64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                # Check.less_than_or_equal_to(max_value=22.0),
            ],
        ),
        "HomeScore": Column(
            pandas_dtype=PandasDtype.Int64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                # Check.less_than_or_equal_to(max_value=168.0),
            ],
        ),
        "AwayScore": Column(
            pandas_dtype=PandasDtype.Int64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                # Check.less_than_or_equal_to(max_value=168.0),
            ],
        ),
        "PossCount": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                # Check.less_than_or_equal_to(max_value=292.0),
            ],
            nullable=True,
        ),
        "SecSinceLastPlay": Column(
            pandas_dtype=PandasDtype.Int64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                # Check.less_than_or_equal_to(max_value=129.0),
            ],
        ),
        "Eventmsgactiontype": Column(
            pandas_dtype=PandasDtype.Int64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                Check.less_than_or_equal_to(max_value=110.0),
            ],
        ),
    },
    # index=Index(
    #     pandas_dtype=PandasDtype.Int64,
    #     checks=[
    #         Check.greater_than_or_equal_to(min_value=0.0),
    #         Check.less_than_or_equal_to(max_value=7017726.0),
    #     ],
    #
    #     coerce=False,
    #     name=None,
    # ),
    coerce=True,
    strict=False,
    name=None,
)

# processed_schema = play_schema.add_columns()
