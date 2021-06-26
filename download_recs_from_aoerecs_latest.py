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

def get_pag():
    for n_box in np.arange(1,8):
        # print(n_box)
        box='/html/body/div/div/main/div[2]/div[3]/div/div['+str(n_box)+']'
        game_name=driver.find_element_by_xpath(xpath=box+'/div/div[1]/h5/div[2]').text
        if game_name=='Random Map TG 1v1 on Arabia':
            #get name:
            name1 = driver.find_element_by_xpath(xpath=box+'/div/div[2]/div/table/tbody[1]/tr/td[1]/span[2]/a').text
            #get civs:
            civ1 = driver.find_element_by_xpath(xpath=box+'/div/div[2]/div/table/tbody[1]/tr/td[2]/a').text
            #get elo:
            elo1 = driver.find_element_by_xpath(xpath=box+'/div/div[2]/div/table/tbody[1]/tr/td[3]').text
            
            name2 = driver.find_element_by_xpath(xpath=box+'/div/div[2]/div/table/tbody[2]/tr/td[1]/span[2]/a').text
            civ2 = driver.find_element_by_xpath(xpath=box+'/div/div[2]/div/table/tbody[2]/tr/td[2]/a').text
            elo2 = driver.find_element_by_xpath(xpath=box+'/div/div[2]/div/table/tbody[2]/tr/td[3]').text
            
            #click to get linkon box
            driver.find_element_by_xpath(box+'/div/div[2]/header/div/div/div/button[4]').click()
            link=driver.find_element_by_xpath(box+'/div/div[2]/div/table/tbody/tr[1]/td[1]/a').get_attribute("href")
            # print(civ1,elo1,civ2,elo2,link)
            data.append((name1,civ1,elo1,name2,civ2,elo2,link))
    return data


link='https://aoe2recs.com/latest/100'
PATH_DRIVER ='C:/Users/ferchi/Desktop/proyecto age/driver selenium/chromedriver'

chrome_options = Options()
chrome_options.add_argument("--headless")

driver = webdriver.Chrome(executable_path = PATH_DRIVER, options = chrome_options)

driver.get(link)
time.sleep(5)

data=[]

for n_pag in tqdm(np.arange(1,10)):
    try:
        get_pag()
        print(n_pag)
    except:
        print('No se encontro en página', n_pag)
        pass
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
            next_page_button=driver.find_element_by_xpath('/html/body/div/div/main/div[2]/div[1]/button['+str(n_button)+']')
            actions.move_to_element(next_page_button).click().perform()
            keep_trying=False   
            time.sleep(5)
        except:
            time.sleep(0.5)
    
    

print('Partidas descargadas: ',len(data))



