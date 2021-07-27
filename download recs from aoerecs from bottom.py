# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 00:56:04 2021

@author: ferchi
"""

import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time 
from selenium.webdriver import ActionChains
from tqdm import tqdm
import pandas as pd
import wget
from multiprocessing.pool import ThreadPool
import requests
import os

counter=1
encontrado=False
def get_pag():
    links=[]
    for n_box in np.arange(1,9):
        # print(n_box)
        box='/html/body/div/div/main/div[2]/div[2]/div[3]/div/div['+str(n_box)+']'
        name1=1
        name2=2
        try:
            #get name:
            name1 = driver.find_element_by_xpath(xpath=box+'/div/div[2]/div/table/tbody[1]/tr/td[1]/span[2]/a').text
            name2 = driver.find_element_by_xpath(xpath=box+'/div/div[2]/div/table/tbody[2]/tr/td[1]/span[2]/a').text
        except:
            pass
        try:
            #get civs:
            civ1 = driver.find_element_by_xpath(xpath=box+'/div/div[2]/div/table/tbody[1]/tr/td[2]/a').text
            #get elo:
            elo1 = driver.find_element_by_xpath(xpath=box+'/div/div[2]/div/table/tbody[1]/tr/td[3]').text
            
            civ2 = driver.find_element_by_xpath(xpath=box+'/div/div[2]/div/table/tbody[2]/tr/td[2]/a').text
            elo2 = driver.find_element_by_xpath(xpath=box+'/div/div[2]/div/table/tbody[2]/tr/td[3]').text
        
            #click to get linkon box
            driver.find_element_by_xpath(box+'/div/div[2]/header/div/div/div/button[4]').click()
            links.append(driver.find_element_by_xpath(box+'/div/div[2]/div/table/tbody/tr[1]/td[1]/a').get_attribute("href"))
            data.append((name1,civ1,elo1,name2,civ2,elo2))
        except:
            pass
    return links
            # print(civ1,elo1,civ2,elo2,link)


link='https://aoe2recs.com/search'
PATH_DRIVER ='C:/Users/ferchi/Desktop/proyecto age/driver selenium/chromedriver'

chrome_options = Options()
chrome_options.add_argument("--headless")

driver = webdriver.Chrome(executable_path = PATH_DRIVER, options = chrome_options)

driver.get(link)
time.sleep(5)
# driver.refresh()

keys=('Arabia',1)

t_espera_chico=0.1
t_espera_grande=0.2

#select arabia
driver.find_element_by_xpath('/html/body/div/div/main/div[2]/div[1]/div[1]/div/div/div/div[1]/div/div/input').click()
time.sleep(t_espera_chico)
driver.find_element_by_xpath('/html/body/div/div/main/div[2]/div[1]/div[1]/div/div/div/div[1]/div/div/input').send_keys(keys[0])
time.sleep(t_espera_grande)

#select 1v1
driver.find_element_by_xpath('/html/body/div/div/main/div[2]/div[1]/div[1]/div/div/div/div[2]/div/div').click()
time.sleep(t_espera_chico)
driver.find_element_by_xpath('/html/body/div[2]/div[3]/ul/li[2]').click()
time.sleep(t_espera_grande)

#select random map
driver.find_element_by_xpath('/html/body/div/div/main/div[2]/div[1]/div[1]/div/div/div/div[3]/div/div').click()
time.sleep(t_espera_chico)
driver.find_element_by_xpath('/html/body/div[2]/div[3]/ul/li[2]').click()
time.sleep(t_espera_grande)

#select ranked
driver.find_element_by_xpath('/html/body/div/div/main/div[2]/div[1]/div[1]/div/div/div/div[5]/div/div').click()
time.sleep(t_espera_chico)
driver.find_element_by_xpath('/html/body/div[2]/div[3]/ul/li[2]').click()
time.sleep(t_espera_grande)

#select DE 1
driver.find_element_by_xpath('/html/body/div/div/main/div[2]/div[1]/div[4]/div[1]/div/div/div/div/div').click()
time.sleep(t_espera_chico)
driver.find_element_by_xpath('/html/body/div[2]/div[3]/ul/li[4]').click()
time.sleep(t_espera_grande)

# #select DE 2
# driver.find_element_by_xpath('/html/body/div/div/main/div[2]/div[1]/div[4]/div[2]/div/div/div/div/div').click()
# time.sleep(0.5)
# driver.find_element_by_xpath('/html/body/div[2]/div[3]/ul/li[7]').click()
# time.sleep(5)

#go to last page
keep_trying=True
while keep_trying:
    try:
        driver.find_element_by_xpath('/html/body/div/div/main/div[2]/div[2]/div[1]/button[9]').click()
        time.sleep(t_espera_chico)
        keep_trying=False   
    except:
        time.sleep(1)  
    


keep_trying=True
while keep_trying:
    try:
        print(driver.find_element_by_xpath('/html/body/div/div/main/div[2]/div[2]/h3').text)
        keep_trying=False   
    except:
        time.sleep(1)  

data=[]

def download_url(link,file_name):
    r = requests.get(link, stream=True)
    with open(file_name, 'wb') as f:
        for chunk in r:
            f.write(chunk)

column_names=('name1','civ1','elo1','name2','civ2','elo2')

N_pags=4000
for n_pag in tqdm(np.arange(1,N_pags)):
    links=get_pag()
    cantidad_links=len(links)
    file_names=[]
    for n_link in range(cantidad_links):
        file_name='C:/Users/ferchi/Desktop/proyecto age/save files/'+'{:05}'.format(n_link+counter)+'.aoe2record'
        link=links[n_link]
        download_url(link,file_name)
    counter+=cantidad_links
    actions = ActionChains(driver)
    time.sleep(0.5)
    keep_trying=True   
    df=pd.DataFrame(data,columns=column_names)
    if n_pag%100==0:
        print('Partidas descargadas: ',len(data))
    df.to_csv('C:/Users/ferchi/Desktop/proyecto age/elo index.csv')
    while keep_trying:
        try:                                               
            next_page_button=driver.find_element_by_xpath('/html/body/div/div/main/div[2]/div[2]/div[1]/button[1]')
            actions.move_to_element(next_page_button).click().perform()
            keep_trying=False   
            time.sleep(2)
        except:
            time.sleep(0.5)  
         


