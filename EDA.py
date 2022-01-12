
import pandas as pd
import os
import bs4
from bs4 import BeautifulSoup
import numpy as np
import sklearn
import matplotlib as mpl
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns


file_name='tFIFA 20 + Statistical data 2020.csv'
df=pd.read_csv(file_name)

print(df.columns)

bins=[0,50,60,70,80,85,90,93]
df['passing groups']=pd.cut(df['PAS'],bins)
df['shooting group']=pd.cut(df['SHO'],bins)
df2=df[df.tots==1]

fig, ax = plt.subplots(1, 2, figsize=(10, 3), sharey=False)





df2.groupby('special cards').tots.sum().plot(kind='barh',ax=ax[0] ,color='salmon')  #kind=line is used in order to obtain a line plot
for c in ax[0].containers:
    labels = [f'{((v.get_width()).astype(np.int64))}' for v in c]
    ax[0].bar_label(c, labels=labels, label_type='edge')
ax[0].set_ylabel('sepcial cards')
ax[0].set_xlabel('amount of TOTS players')
ax[0].set_title('Special cards for TOTS players')





#ax = ax[1].axes()
ax[1].scatter(df.goals, df.assists) #drawing all dfemons
ax[1].set_xlabel('goals')
ax[1].set_ylabel('assists')
ax[1].set_title('Players goals and assists')
ax[1].scatter(df.goals[df.tots==1], df.assists[df.tots==1], c='red') #drawing only is_legendary in red

plt.show()



fig, ar = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
df.groupby('shooting group').mean()[['goals','assists']].plot(kind='line',ax=ar[0]).legend(loc='upper left',ncol=1)  #kind=line is used in order to obtain a line plot
#for c in ax1[1].containers:
    #labels = [f'{(v.get_height()):.1f}' for v in c]
    #ax1[1].bar_label(c, labels=labels, label_type='edge')
ar[0].set_ylabel('Mean goals and assists')
ar[0].set_xlabel('FIFA SHO/PAS Rating')
ar[0].set_title('Mean goals and assists by passing and shooting rating')


b=df.groupby(['POS']).mean()[['goals','assists','xGBuildup','xGChain']]
b.plot(ax=ar[1],style=['+-','o-','.--','+--']).legend(loc='upper right', ncol=2)
ar[1].set_ylabel('Mean stats')
ar[1].set_xlabel('Player Position')
ar[1].set_title('Mean stasts by player position')
plt.show()









g = sns.catplot(

    x='POS', 
    data=df,
    kind='count', 
    hue='tots',
    palette=["cyan", "lightcoral"], 
    height=3, 
    aspect=2.5,
    legend=True,

    ).set_axis_labels('position', 'tots')
g.ax.legend(labels=['NON TOTS','TOTS'])
g.ax.set_title('TOTS Distribution by player position')
ax = g.facet_axis(0, 0)
for c in ax.containers:
    
    labels = [f'{(int(v.get_height()))}' for v in c]
    ax.bar_label(c, labels=labels, label_type='edge')

plt.show()

tots_Sum=df['tots'].value_counts()
i = [0]
def tots_val(val):
    a  = tots_Sum[i[0]]
    i[0] += 1
    return a


tots_Sum.plot.pie(subplots=True,autopct=tots_val,labels=['NON-TOTS','TOTS'],title='TOTS Distribution')

plt.show()




g = sns.catplot(
    x='Team position', 
    data=df,
    kind='count', 
    hue='tots',
    palette=["cyan", "lightcoral"], 
    height=3, 
    aspect=2.5,
    legend=True,

    ).set_axis_labels('Team position', 'amount of tots cards')
g.ax.legend(labels=['NON TOTS','TOTS'])
g.ax.set_title('TOTS Distribution by team position')

ax = g.facet_axis(0, 0)
for c in ax.containers:
    
    labels = [f'{((v.get_height()).astype(np.int64))}' for v in c]
    

    ax.bar_label(c, labels=labels, label_type='edge')
  

plt.show()