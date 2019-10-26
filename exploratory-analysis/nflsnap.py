import numpy as np 
import pandas as pd
import sqlalchemy
import matplotlib.pyplot as plt
from data_utilities import readSQL
from settings import ENGINE_URL


engine = sqlalchemy.engine_from_config(
    ENGINE_URL, prefix='FOOTBALL_TEST.')
query = readSQL.readSQL('data/backtest.sql', query_name=False)

tmp = pd.read_sql(query, engine)

tmp1 = tmp.loc[tmp['Version'] == 0, :].drop(columns=['Season', 'Version'])
tmp2 = tmp.loc[tmp['Version'] == 1, :].drop(columns=['Season', 'Version'])

tmpp = pd.merge(tmp1, tmp2, how='outer', 
                on=['GameId', 'FranchiseId', 'OpponentId', 'PlayerId', 'Unit', 'Role'])
tmpp1 = tmpp.loc[tmpp['GameId'].isin([13767,13762,13832]),:]

tmpp1 = tmpp1.assign(res_x=tmpp1.AdjPred_x - tmpp1.Ratio_x, 
                     res_y=tmpp1.AdjPred_y - tmpp1.Ratio_y)
tmpp1 = tmpp1.assign(resdiff=(tmpp1.res_x.abs() - tmpp1.res_y.abs()).abs())
tmpp1 = tmpp1.assign(prediff=(tmpp1.AdjPred_x - tmpp1.AdjPred_y).abs())
