import numpy as np 
import pandas as pd
import sqlalchemy
from settings import ENGINE_URL


engine = sqlalchemy.engine_from_config(
    ENGINE_URL, prefix='FOOTBALL_TEST.')

tmp = pd.read_sql('select * from Model_NFL_RoleSnap', engine)

tmp1 = tmp.loc[tmp['Version'] == 0 & tmp['Season'] == 2019, :]
