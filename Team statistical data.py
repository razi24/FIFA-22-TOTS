import pandas as pd
import csv
import os
import bs4
from bs4 import BeautifulSoup
import numpy as np
from pandas.core import series

from requests.api import head

import scipy as sc
import requests
import json
from unidecode import unidecode

#Definnig the leagues and the seasons
base_url = 'https://understat.com/league'
leagues = [ 'la-liga','premier-league', 'bundesliga', 'serie-a', 'ligue-1']
fifa_leagues= ['ESP 1', 'ENG 1', 'GER 1', 'ITA 1', 'FRA 1']
seasons = ['2019', '2020', '2021']


for i in range(3):
    file_name="League Tabels " + seasons[i]+ ".csv"
    f=open(file_name,'w')
    csv_writer=csv.writer(f)

    for j in range(5):
        #url="https://www.skysports.com/" + leagues[j] + "/" + seasons[i] 
        url="https://www.skysports.com/" + leagues[j] + "-table" + "/" + seasons[i] 
        page= requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        
        data=[]
        
        
        if(j==0):
            t_head=soup.find('thead')
        
        
            for th in t_head.find_all('th'):
                data.append(th.text.strip()) 


            data.append('League')    





        csv_writer.writerow(data)
        t_body=soup.find('tbody')



        for tr in t_body.find_all('tr'):
            data=[]
            simple_data=[]


            for td in tr.find_all('td'):
            
                
                text2=td.text.strip()
                text2=text2.replace('\n',' ')
                data.append(text2)         
            


            if data:
            #print("inserting data:{}".format(','.join(data)))
                data.append(fifa_leagues[j])
                csv_writer.writerow(data)

    
         

f.close()


  
    
    











