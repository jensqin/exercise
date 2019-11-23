import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 

tips = sns.load_dataset('tips')
sns.relplot(
    "total_bill", y="tip", hue='sex', data=tips,
    style='sex'
)
sns.kdeplot(tips['total_bill'], tips['tip'], cmap='Reds', shade=True)
