import pandas as pd
import os
import bs4
from bs4 import BeautifulSoup
import numpy as np
from pandas.core import series
import scipy as sc
import requests

from difflib import SequenceMatcher

# The function retures the similarity index between two names
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


# Reading the files and creating the data frames to merge 
years=['20','21','22']



for x in range(3):
     file_name='FIFA'+ years[x]+ '.csv'
     file_name1='Player Statistic 20' +years[x] + '.csv'
     file_name2='League Tabels 20'+years[x] + '.csv'
     df_FIFA=pd.read_csv(file_name)
     df_Stat=pd.read_csv(file_name1)
     df_league_stat=pd.read_csv(file_name2)


     # Definnig leagues and seasons
     leagues = ['ESP 1', 'ENG 1', 'GER 1', 'ITA 1', 'FRA 1']
     seasons = ['2019', '2020', '2021'] 

     # Creating TOTS label 
     tots_label=[0 for i in range(df_FIFA.shape[0])]
     positions=['GK','Defender','Midfielder','Attacker']
     df_FIFA['tots']=tots_label
     df_FIFA.loc[df_FIFA['tots']].loc[df_FIFA['Version'] == 'tots']=1
     tots_player=list(df_FIFA.loc[df_FIFA['Version']=='tots']['Player'])
     tots_player_team=list(df_FIFA.loc[df_FIFA['Version']=='tots']['Team'])



     for i in range (df_FIFA.shape[0]) :

          if((df_FIFA.loc[i,'Player']  in tots_player)&(df_FIFA.loc[i,'Team']  in tots_player_team)):
               df_FIFA.loc[i,'tots']=1
          if(df_FIFA.loc[i,'POS'] in ['CB','RB','LB','LWB','RWB']):
               df_FIFA.loc[i,'POS']= 'Defender' 
          elif(df_FIFA.loc[i,'POS'] in ['CM','CDM','RM','LM','CAM']):
               df_FIFA.loc[i,'POS']= 'Midfielder'
          elif(df_FIFA.loc[i,'POS'] in ['ST','CF','RW','RF','LW','LF']):
               df_FIFA.loc[i,'POS']= 'Attacker'
          else:
               df_FIFA.loc[i,'POS']= 'GK'     







     # creating df for 5 big leagues
     df_major_leagues=df_FIFA.loc[df_FIFA['League'].isin(leagues)]
     df_copy_major=df_major_leagues.copy()

     # creating special cards column
     df_copy_major['special cards']=[i for i in range(df_copy_major.shape[0])]
     df_copy_major.set_index(pd.Index(list(df_copy_major['special cards'])),inplace=True)
     for i in range(df_copy_major.shape[0]):
          df_copy_major.loc[i,'special cards']=list(df_copy_major['Player']).count(df_copy_major.loc[i,'Player'])-1



     # dropping duplicated cards for players 
     df_copy_major.drop_duplicates(subset=['Player','Team'],keep='last',inplace=True)
     df_copy_major.reset_index(inplace=True)


     #removing signs from player names


     # matching the Teams Names's between the players data frames
     list_team_fifa=list(df_copy_major.Team.unique())


     list_stat=list(df_Stat.Team.unique())

     combained_list=list(set(list_stat+list_team_fifa))
     differnces=list(set(combained_list)-set(list_team_fifa))

     for i in range(len(differnces)):
          league_i=df_Stat.loc[df_Stat['Team']==differnces[i]].copy()
          league_i.reset_index(inplace=True)
          league_2=league_i.loc[0,'League']
          df_team=list((df_copy_major.loc[df_copy_major['League']==league_2])['Team'].unique())
     
          range_value=0.1
          new_value='a'
          for j in range(len(df_team)):
               #print(differnces[i],df_team[j])
               check_value= similar(differnces[i],df_team[j])
               if(check_value>range_value):
                    range_value=check_value
                    new_value=df_team[j]

          #print('name team change :', differnces[i],'----->',new_value)
          df_Stat['Team'].replace(differnces[i],new_value,inplace=True)   



     # matching the Teams Names's between the two data frames
     list_team_fifa=list(df_copy_major.Team.unique())


     list_stat=list(df_league_stat.Team.unique())

     combained_list=list(set(list_stat+list_team_fifa))
     differnces=list(set(combained_list)-set(list_team_fifa))

     for i in range(len(differnces)):
          league_i=df_league_stat.loc[df_league_stat['Team']==differnces[i]].copy()
          league_i.reset_index(inplace=True)
          league_2=league_i.loc[0,'League']
          df_team=list((df_copy_major.loc[df_copy_major['League']==league_2])['Team'].unique())
          
          range_value=0.1
          new_value='a'
          for j in range(len(df_team)):
               #print(differnces[i],df_team[j])
               check_value= similar(differnces[i],df_team[j])
               if(check_value>range_value):
                    range_value=check_value
                    new_value=df_team[j]

          print('name team change :', differnces[i],'----->',new_value)
          df_league_stat['Team'].replace(differnces[i],new_value,inplace=True)   







     #replacing problamtic names for players

     df_combained_checked=pd.merge(df_copy_major,df_Stat)


     problamatic_players=list(set(list(df_Stat['Player']))-set(list(df_combained_checked['Player'])))

     #print(len(problamatic_players))


     for i in range(len(problamatic_players)):



          team_i=df_Stat.loc[df_Stat['Player']==problamatic_players[i]].copy()
          team_i.reset_index(inplace=True)
          team_2=team_i.loc[0,'Team']
          
          df_player=list((df_copy_major.loc[df_copy_major['Team']==team_2])['Player'].unique())
          df_player2=list(df_Stat['Player'].unique())

          range_value=0.51
          new_value='a'

          for j in range(len(df_player)):
               check_value=similar(problamatic_players[i],df_player[j])
               if((check_value>range_value) & (not(df_player[j] in df_player2))):
                    range_value=check_value
                    new_value=df_player[j]

     
          if(new_value != 'a'):
               df_Stat['Player'].replace(problamatic_players[i],new_value,inplace=True)
               #print('name change :', problamatic_players[i],'----->',new_value)
          #else:
               #print("Can't find it in the fifa data frame check :" ,problamatic_players[i])        







     #combining the stat and the fifa data frames


     #merge stat and fifa
     df_combained1=pd.merge(df_copy_major,df_Stat)
     problamatic_players2=list(set(list(df_Stat['Player']))-set(list(df_combained1['Player'])))
     df_combained1.drop(labels=['Unnamed: 0','id','position','BIN','index','index'],axis=1,inplace=True)


     #merge stat and fifa with team stat
     df_combained2=pd.merge(df_combained1,df_league_stat)
     df_combained2.sort_values(by='OVR',ascending=False,inplace=True)
     df_combained2.rename(columns={ '#' :'Team position','F' :'Team Goals for','A':'Team Goals against'},inplace=True)
     df_combained2.drop(labels=['GD','Pts','Last 6','Pl','W','D','L'],axis=1,inplace=True)
     #print(len(problamatic_players2))



     #encoding labels for league and pos
     df_combained2.loc[:, ["League",'POS']] = df_combained2.loc[:, ["League",'POS']].astype('category')
     df_combained2.loc[:,"League-label"] = df_combained2["League"].cat.codes +1
     df_combained2.loc[:,"Pos-label"] = df_combained2["POS"].cat.codes +1






     file_name1="tFIFA " + years[x]+ ' + ' 'Statistical data 20' +years[x]+ ".csv"
     f=open(file_name1,'w')
     df_combained2.to_csv(path_or_buf=file_name1)
     f.close()
