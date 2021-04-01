import pandas as pd
import sqlalchemy
from bla_python_db_utilities.parser import parse_sql

from utils import ENGINE_CONFIG

engine = sqlalchemy.create_engine(
    "mysql+pymysql://zqin:H3re15@newP@ssBB@ll123!@127.0.0.1:3309/NBA"
)

sqls = parse_sql("sql/sportradar.sql")
df = pd.read_sql(sqls["sportradar"], engine)

