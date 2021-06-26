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
from selenium.webdriver.support.ui import Select
from tqdm import tqdm
import pandas as pd

N_pags=10
counter=0
# data=np.zeros((N_pags*8))
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
            counter+=1
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


keep_trying=True
while keep_trying:
    try:
        print(driver.find_element_by_xpath('/html/body/div/div/main/div[2]/div[2]/h3').text)
        keep_trying=False   
    except:
        time.sleep(1)  

data=[]
for n_pag in tqdm(np.arange(1,N_pags)):
    previous_counter=np.copy(counter)
    links=get_pag()
    for link in links:
        '{:05}'.format(counter)
    #print(n_pag)
    actions = ActionChains(driver)
    time.sleep(0.5)
    if n_pag<4:
        n_button=10
    elif n_pag==4:
        n_button=11
    elif n_pag==5:
        n_button=12
    else:
        n_button=13
    keep_trying=True   
    while keep_trying:
        try:                                               
            next_page_button=driver.find_element_by_xpath('/html/body/div/div/main/div[2]/div[2]/div[1]/button['+str(n_button)+']')
            actions.move_to_element(next_page_button).click().perform()
            keep_trying=False   
            time.sleep(2)
        except:
            time.sleep(0.5)  
column_names=('name1','civ1','elo1','name2','civ2','elo2')
df=pd.DataFrame(data)
print('Partidas descargadas: ',len(data))



