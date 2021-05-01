import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series

from nbastats.schema.example import InputSchema, OutputSchema


@pa.check_types
def transform(df: DataFrame[InputSchema]) -> DataFrame[OutputSchema]:
    return df.assign(revenue=100.0)


df = pd.DataFrame(
    {
        "year": ["2001", "2002", "2003"],
        "month": ["3", "6", "12"],
        "day": ["200", "156", "365"],
    }
)
transform(df)

invalid_df = pd.DataFrame(
    {
        "year": ["2001", "2002", "1999"],
        "month": ["3", "6", "12"],
        "day": ["200", "156", "365"],
    }
)
transform(invalid_df)
