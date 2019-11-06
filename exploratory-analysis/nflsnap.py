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

tmpp2 = (
    tmpp.loc[tmpp['GameId'].isin([13767,13762,13832]),:]
    .groupby(['GameId','FranchiseId','PlayerId','Unit', 'Status_x', 'Status_y'])
    .agg({'Ratio_x': 'sum', 'Ratio_y': 'sum', 'AdjPred_x': 'sum', 'AdjPred_y': 'sum'})
    .reset_index()
)

tmpp2 = tmpp2.assign(res_x=tmpp2.AdjPred_x - tmpp2.Ratio_x,
                     res_y=tmpp2.AdjPred_y - tmpp2.Ratio_y,
                     prediff=(tmpp2.AdjPred_x - tmpp2.AdjPred_y).abs())
tmpp2.loc[tmpp2['GameId'] == 13832, :].sort_values('prediff',ascending=False).head(10)
tmpp2.loc[tmpp2['GameId'] == 13762, :].sort_values('prediff',ascending=False).head(10)
tmpp2.loc[tmpp2['GameId'] == 13767, :].sort_values('prediff',ascending=False).head(10)
