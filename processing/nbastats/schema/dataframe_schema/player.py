from pandas import Timestamp
from pandera import (
    DataFrameSchema,
    Column,
    Check,
    Index,
    MultiIndex,
    PandasDtype,
)

player_schema = DataFrameSchema(
    columns={
        "PlayerId": Column(
            pandas_dtype=PandasDtype.Int64,
            checks=[Check.greater_than_or_equal_to(min_value=2.0),],
            nullable=False,
            allow_duplicates=True,
            coerce=False,
            required=True,
            regex=False,
        ),
        "Name": Column(
            pandas_dtype=PandasDtype.String,
            checks=None,
            nullable=True,
            allow_duplicates=True,
            coerce=False,
            required=True,
            regex=False,
        ),
        "Dob": Column(
            pandas_dtype=PandasDtype.DateTime,
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=Timestamp("1900-01-01 00:00:00")
                ),
            ],
            # fix the null value of birth date
            # nullable=False,
            nullable=True,
            allow_duplicates=True,
            coerce=False,
            required=True,
            regex=False,
        ),
        "DraftYear": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=1963.0),
                Check.less_than_or_equal_to(max_value=2020.0),
            ],
            nullable=True,
            allow_duplicates=True,
            coerce=False,
            required=True,
            regex=False,
        ),
        "DraftRound": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                Check.less_than_or_equal_to(max_value=8.0),
            ],
            nullable=True,
            allow_duplicates=True,
            coerce=False,
            required=True,
            regex=False,
        ),
        "DraftNumber": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[
                Check.greater_than_or_equal_to(min_value=1.0),
                Check.less_than_or_equal_to(max_value=165.0),
            ],
            nullable=True,
            allow_duplicates=True,
            coerce=False,
            required=True,
            regex=False,
        ),
        "Country": Column(
            pandas_dtype=PandasDtype.String,
            checks=None,
            nullable=True,
            allow_duplicates=True,
            coerce=False,
            required=True,
            regex=False,
        ),
        "College": Column(
            pandas_dtype=PandasDtype.String,
            checks=None,
            nullable=True,
            allow_duplicates=True,
            coerce=False,
            required=True,
            regex=False,
        ),
        "Position": Column(
            pandas_dtype=PandasDtype.String,
            checks=None,
            nullable=True,
            allow_duplicates=True,
            coerce=False,
            required=True,
            regex=False,
        ),
        "HeightInInches": Column(
            pandas_dtype=PandasDtype.Float64,
            checks=[Check.greater_than_or_equal_to(min_value=63.0),],
            nullable=True,
            allow_duplicates=True,
            coerce=False,
            required=True,
            regex=False,
        ),
    },
    index=Index(
        pandas_dtype=PandasDtype.Int64,
        checks=[Check.greater_than_or_equal_to(min_value=0.0),],
        nullable=False,
        coerce=False,
        name=None,
    ),
    coerce=True,
    strict=False,
    name=None,
)
