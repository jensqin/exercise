import pandas as pd
from sklearn import linear_model
import tensorflow as tf
# from tensorflow.keras import layers
from tensorflow import keras
import seaborn as sns

ftsoff = pd.read_feather("~/efs/zqin/feather/ftsoff.feather")
# tmp = (ftsoff
#        .loc[(ftsoff.Role == "Pass") & (ftsoff.GamePosition == "QB")]
#        .loc[ftsoff.Status == "A"]
#        .loc[ftsoff.PassAtt < 100]
#        .loc[ftsoff.Season > 2014]
#       )
# sns.lmplot(x="PassAtt",y="Ratio",hue="Season",data=tmp,lowess=True)

# ftsridge = linear_model.Ridge(alpha=1.0)
# ftsridge.fit(X=tmp[["PassAtt", "Completions", "PassInt", "PassRat", "FPTS", "PtsAll"]],
#             y=tmp["Ratio"])

tmp = (ftsoff
      .loc[ftsoff.GamePosition == "QB"]
      )
tmp.loc[:,["GamePosition","Role","PositionSet","Status","SeasonType"]]=(
    tmp.loc[:, ["GamePosition","Role","PositionSet","Status","SeasonType"]].astype('category')
)

tmpp = tmp.loc[:,['Unit', 'GameId', 'PlayerId', 'FranchiseId', 'OpponentId', 'Season',
       'Week', 'GamePosition', 'Status', 'Attend',
       'PositionSet', 'SeasonType', 'Position', 'FantasyPosition', 'PassAtt',
       'Completions', 'PassYds', 'PassTD', 'PassInt', 'PassRat', 'PassSacks',
       'RushAtt', 'RushYds', 'RushTD', 'Targets', 'Rec', 'RecYds', 'RecTD',
       'Fumbles', 'FPTS', 'PtsAll', 'TeamPassDef', 'TeamPass', 'TeamRun']]
tmpp = tmpp.drop_duplicates()

tmpp1 = tmp.loc[:, ["GameId","PlayerId","Role","Ratio"]]
tmpp1 = tmpp1.pivot_table(index=["GameId","PlayerId"],values="Ratio",columns="Role")

tmpp2 = pd.merge(tmpp, tmpp1, left_on=["GameId","PlayerId"], right_index=True)

x_train = pd.get_dummies(tmpp2[['SeasonType','Status','PassAtt',
       'Completions', 'PassYds', 'PassTD', 'PassInt', 'PassRat', 'PassSacks',
       'RushAtt', 'RushYds', 'RushTD', 'Targets', 'Rec', 'RecYds', 'RecTD',
       'Fumbles', 'FPTS', 'PtsAll', 'TeamPassDef', 'TeamPass', 'TeamRun']], 
                         drop_first=True).to_numpy()

y_train = tmpp2[["Pass","Pass Block", "Run", "Run Block"]].to_numpy()

# inputs = tf.keras.Input(shape=(25,), name='fantasy_QB')
# x = layers.Dense(15, activation='relu')(inputs)
# x = layers.Dense(15, activation='relu')(x)
# outputs = layers.Dense(4, activation='softmax')(x)

# ftstf = tf.keras.Model(inputs=inputs, outputs=outputs, name='ftsQB')
# ftstf.compile(loss='categorical_crossentropy',
#               optimizer=tf.keras.optimizers.RMSprop(),
#               metrics=['mse'])
# tffit = ftstf.fit(x_train, y_train,
#                     epochs=5)

kmodel = keras.models.Sequential()
kmodel.add(keras.Input(shape=(25,), name="fantasy_QB"))
kmodel.add(keras.layers.Dense(25, activation="elu", kernel_initializer="he_normal", 
          kernel_regularizer=keras.regularizers.l2(0.01)))
kmodel.add(keras.layers.Dropout(rate=0.1))
kmodel.add(keras.layers.Dense(25, activation="elu", kernel_initializer="he_normal",
          kernel_regularizer=keras.regularizers.l2(0.01)))
kmodel.add(keras.layers.Dropout(rate=0.1))
kmodel.add(keras.layers.Dense(4, activation="softmax", kernel_initializer="he_normal",
          kernel_regularizer=keras.regularizers.l2(0.01)))
kmodel.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(),
              metrics=["mse"])
kfit = kmodel.fit(x_train, y_train, epochs=10)
