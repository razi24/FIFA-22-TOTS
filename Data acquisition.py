import pandas as pd
import os
import bs4
from bs4 import BeautifulSoup
import numpy as np
from pandas.io.formats import style
import scipy as sc
import requests
import csv



#url="https://www.futwiz.com/en/fifa20/players?page=0"
#pages=955
#url="https://www.futwiz.com/en/fifa21/players?page=0"
#pages=907
url="https://www.futwiz.com/en/fifa22/players?page=0"
pages=753
page= requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
file_name='FIFA22.csv'
f=open(file_name,'w')
csv_writer=csv.writer(f)


# Writing the headers of the table
data=['Player','Team','League','Version']
t_head=soup.find('tr',class_="head")
for th in t_head('td'):
    data.append(th.text.strip()) 
    
del data[4]
del data[4]


csv_writer.writerow(data)


# Writing the data from all pages 

for i in range(753):
    
 
    for tr in soup.find_all('tr',class_="table-row"):
        data=[]
        simple_data=[]


        for td in tr.find_all('td'):
        
            if( td.find_all('p')):
                p1=td.find('p',class_="name")
                data.append(p1.text.strip())
                p2=p1.find('a')
                


                
                p4=td.find('p',class_="team")
                    
            
                for x in p4.find_all('a'):
                    data.append(x.text)
                    
                    
                
                
                
                

            else:
                if(td.find('span',class_='stat')):
                    x=td.find('span',class_='stat')
                    data.append(x.text)

                else:
                    if(td.find('div',class_="otherversion22-txt")): # Getting the virsion of the card
                        x7=td.find('div')
                        x7=x7['class'][1]
                        x7=x7[x7.index("-"):]
                        x7=x7[1:]
                        data.append(x7)
                        x8=td.text.strip()
                        data.append(x8)

                    else:


                        text2=td.text.strip()
                        text2=text2.replace('\n',' ')
                        data.append(text2)   

                      
        del data[0]

        if data:
            
            csv_writer.writerow(data)

    # Changing the url to the next page 

    if(i<10):
        url=url[:-1]+str(i+1)
        page= requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser') 
    if((i>=10)&(i<100)):
        url=url[:-2]+str(i+1)
        page= requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser') 
    if(i>=100):
        url=url[:-3]+str(i+1)
        page= requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')               

f.close()
