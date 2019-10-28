import numpy as np 
import pandas as pd 
import settings
import sqlalchemy
from data_utilities import readSQL
import argparse

engine = sqlalchemy.engine_from_config(settings.ENGINE_URL, prefix='BASKETBALL_DEV.')

query = readSQL.readSQL('ncaab_minspl/data/minutes.sql', season='AND Season = 2019')
minutes = pd.read_sql(query['minutes'], engine)
inj = pd.read_sql(query['injury'], engine)
# exempt = pd.read_sql(query['exempt'], engine)

mins.s = (
    minutes
    .fillna(value={'Mins': 0})
    .groupby(['GameId', 'TeamId'])
    .assign(s=minutes.Mins.sum())
)

