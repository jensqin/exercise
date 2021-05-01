import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series


class InputSchema(pa.SchemaModel):
    year: Series[int] = pa.Field(gt=2000, coerce=True)
    month: Series[int] = pa.Field(ge=1, le=12, coerce=True)
    day: Series[int] = pa.Field(ge=0, le=365, coerce=True)

    @pa.dataframe_check
    def length_limit(cls, df: pd.DataFrame) -> Series[bool]:
        return len(df["year"]) >= 3


class OutputSchema(InputSchema):
    revenue: Series[float]
