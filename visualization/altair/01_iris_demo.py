import pandas as pd 
import seaborn as sns 
import altair as alt 

iris = sns.load_dataset('iris')
tips = sns.load_dataset('tips')

# Target:
# 1. one, two-dimentional charts
# 2. the third dimentional: color and others
# 3. binning and aggregation
# 4. time series and layers
# 5. interactive plot


# 1D scatter plot
alt.Chart(iris).mark_point().encode(
    alt.X('sepal_length')
)

# 1D stick plot
alt.Chart(iris).mark_tick().encode(
    alt.X('sepal_length')
)

# histogram
alt.Chart(iris).mark_bar().encode(
    alt.X('sepal_length', bin=True, title='sepal_length'),
    alt.Y('count()'),
    alt.Color('petal_length')
)
alt.Chart(iris).mark_bar().encode(
    alt.X('sepal_length', bin=alt.Bin(maxbins=30)),
    alt.Y('count()')
)
alt.Chart(iris).mark_bar().encode(
    alt.X('sepal_length', bin=alt.Bin(maxbins=30)),
    alt.Y('count()'),
    alt.Color('species'),
    alt.Column('species')
)
alt.Chart(iris).mark_bar().encode(
    x=alt.X('sepal_length', bin=alt.Bin(maxbins=30)),
    y=alt.Y('sepal_width'),
    color='mean(petal_length)'
)

# time series
alt.Chart(tips).mark_bar().encode(
    x=alt.X('size:O'),
    y=alt.Y('count()'),
    color=alt.Color('sex')
)
alt.Chart(tips).mark_line().encode(
    x=alt.X('size'),
    y=alt.Y('mean(tip)')
)
alt.Chart(tips).mark_line().encode(
    x=alt.X('size'),
    y=alt.Y('mean(tip)')
)
area = alt.Chart(tips).mark_area(opacity=0.3).encode(
    x=alt.X('size'),
    y=alt.Y('ci0(tip)'),
    y2=alt.Y2('ci1(tip)'),
    color=alt.Color('sex')
).properties(
    width=400
)
lines = alt.Chart(tips).mark_line().encode(
    x=alt.X('size'),
    y=alt.Y('mean(tip)'),
    color=alt.Color('sex')
).properties(
    width=400
)
area + lines

# interactive scatter plot
alt.Chart(iris).mark_point().encode(
    alt.X('sepal_length', title='length'),
    alt.Y('sepal_width', title='width'),
    alt.Color('species')
).interactive()

interval = alt.selection_interval()
alt.Chart(iris).mark_point().encode(
    x='sepal_length',
    y='sepal_width',
    color=alt.condition(interval, 'species', alt.value('lightgrey'))
).properties(
    selection=interval
)

# two interactive plots
baseplot = alt.Chart(iris).mark_point().encode(
    y='sepal_width',
    color=alt.condition(interval, 'species', alt.value('lightgrey')),
    tooltip='petal_length'
).properties(
    selection=interval
)
scatter = baseplot.encode(x='sepal_length') | baseplot.encode(x='petal_length')

hist = alt.Chart(iris).mark_bar().encode(
    x='count()',
    y='species',
    color='species'
).properties(
    width=400,
    height=100
).transform_filter(
    interval
)
scatter & hist
