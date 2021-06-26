# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 18:55:25 2021

@author: ferchi con ayuda de matiu
"""

from matplotlib import pyplot as plt
from mgz.summary import Summary
import numpy as np
from find_cluster_center import find_cluster_center


def get_features(path,plot_objects=False,plot_forests=False):
    '''
    From a given path name get the parameters:
    ELO of players
    players civilizations    
        
    gold tiles center
    stone tiles center
    berries center
    TC locations
    Forests that are close to the TCs's center
    
    Object ids importantes:
    oro: 66
    piedra: 102
    TC:1649
    bayas: 1059 or 59
    
    
    aldeanos: 83 (Hombre) 293 (Mujer)
    reliquia: 285
    ovejas: 1060,594
    boar: 48
    scout:751 (aguila) 448 (scout)
    
    Terrain ids:
    bosque:10,13, 18
    '''
    bosque_ids=(10,13,18,88)
    
    with open(path, 'rb') as data:
        s = Summary(data)
        mapa=s.get_map()
        tiles=mapa['tiles']
        objetos=s.get_objects()['objects']
        ratings=s.get_players()
    plt.close('all')
    obj=np.zeros((120,120))
    
    
    piedras=[]
    oros=[]
    TCs=[]
    bayas=[]
    
    for diccionario in objetos:
        x=int(diccionario['x'])
        y=int(diccionario['y'])
        obj_id=diccionario['object_id']
        if obj_id!=647:#saco los circulos donde pueden spawnear las ovejas, es algo interno del juego
            obj[y,x]=diccionario['object_id']
        if obj_id==66:
            oros.append([y,x])
        if obj_id==102:
            piedras.append([y,x])
        if obj_id==1649:
            TCs.append([y,x])
        if obj_id==1059 or obj_id==59:
            bayas.append([y,x])
    
    oros=np.array(oros)
    piedras=np.array(piedras)
    coord_TCs=np.array(TCs)
    bayas=np.array(bayas)
    
    coord_oros=find_cluster_center(oros, 8)
    coord_piedras=find_cluster_center(piedras, 6)
    coord_bayas=find_cluster_center(bayas, 2)
    
    axis=[0,120,0,120]
    if plot_objects==True:
        plt.figure(figsize=(8,8))
        plt.imshow(obj,extent=axis, interpolation='none', aspect='equal')
        plt.scatter(coord_oros[:, 1], 120-coord_oros[:, 0], c='yellow', s=100, alpha=0.6)
        plt.scatter(coord_piedras[:, 1], 120-coord_piedras[:, 0], c='grey', s=100, alpha=0.6)
        plt.scatter(coord_bayas[:, 1], 120-coord_bayas[:, 0], c='red', s=100, alpha=0.6)
    
    bosques=[]
    terreno=np.zeros((120,120))
    d_bosque_TC=30#maxima distancia entre bosque y tc para que sea uno de los bosques iniciales
    # elevacion=np.zeros((120,120))
    for diccionario in tiles:
        x=int(diccionario['x'])
        y=int(diccionario['y'])
        terr_id=diccionario['terrain_id']
        if terr_id in bosque_ids:
            if ((y-coord_TCs[0,0])**2+(x-coord_TCs[0,1])**2)**0.5<d_bosque_TC or ((y-coord_TCs[1,0])**2+(x-coord_TCs[1,1])**2)**0.5<d_bosque_TC:
                bosques.append([y,x])
        terreno[y,x]=diccionario['terrain_id']
        # elevacion[x,y]=diccionario['elevation']
    
    coord_bosques=find_cluster_center(bosques, 6)
    
    if plot_forests==True:
        plt.figure(figsize=(8,8))
        plt.imshow(terreno,extent=axis, interpolation='none', aspect='equal')
        plt.scatter(coord_bosques[:, 1], 120-coord_bosques[:, 0], c='green', s=100, alpha=0.5)

    return ratings[0]['name'],ratings[0]['civilization'],ratings[0]['position'],ratings[0]['winner'],ratings[1]['name'],ratings[1]['civilization'],ratings[1]['position'],ratings[1]['winner'],coord_oros,coord_piedras,coord_bayas,coord_bosques

if __name__ == '__main__':        
    folder='C:/Users/ferchi/Desktop/proyecto age/aoe recs/'
    filename='AgeIIDE_Replay_100014172.aoe2record'
    filename='game.aoe2record'
    filename='AgeIIDE_Replay_100552757.aoe2record'
    # filename='MP Replay v101.101.36202.0 @2020.04.03 202628 (2).aoe2record'
    # filename='MP Replay v101.101.47820.0 @2021.06.07 220925 (2).aoe2record'
    path=folder+filename
    plot_objects=True
    plot_forests=True

    bosque_ids=(10,13,18,88)
    
    with open(path, 'rb') as data:
        s = Summary(data)
        platform=s.get_platform()
        mapa=s.get_map()
        tiles=mapa['tiles']
        objetos=s.get_objects()['objects']
        ratings=s.get_players()
    plt.close('all')
    obj=np.zeros((120,120))
    
    
    piedras=[]
    oros=[]
    TCs=[]
    bayas=[]
    
    for diccionario in objetos:
        x=int(diccionario['x'])
        y=int(diccionario['y'])
        obj_id=diccionario['object_id']
        if obj_id!=647:#saco los circulos donde pueden spawnear las ovejas, es algo interno del juego
            obj[y,x]=diccionario['object_id']
        if obj_id==66:
            oros.append([y,x])
        if obj_id==102:
            piedras.append([y,x])
        if obj_id==1649:
            TCs.append([y,x])
        if obj_id==1059 or obj_id==59:
            bayas.append([y,x])
    
    oros=np.array(oros)
    piedras=np.array(piedras)
    coord_TCs=np.array(TCs)
    bayas=np.array(bayas)
    
    coord_oros=find_cluster_center(oros, 8)
    coord_piedras=find_cluster_center(piedras, 6)
    coord_bayas=find_cluster_center(bayas, 2)
    
    axis=[0,120,0,120]
    if plot_objects==True:
        plt.figure(figsize=(8,8))
        plt.imshow(obj,extent=axis, interpolation='none', aspect='equal')
        plt.scatter(coord_oros[:, 1], 120-coord_oros[:, 0], c='yellow', s=100, alpha=0.6)
        plt.scatter(coord_piedras[:, 1], 120-coord_piedras[:, 0], c='grey', s=100, alpha=0.6)
        plt.scatter(coord_bayas[:, 1], 120-coord_bayas[:, 0], c='red', s=100, alpha=0.6)
    
    bosques=[]
    terreno=np.zeros((120,120))
    d_bosque_TC=30#maxima distancia entre bosque y tc para que sea uno de los bosques iniciales
    # elevacion=np.zeros((120,120))
    for diccionario in tiles:
        x=int(diccionario['x'])
        y=int(diccionario['y'])
        terr_id=diccionario['terrain_id']
        if terr_id in bosque_ids:
            if ((y-coord_TCs[0,0])**2+(x-coord_TCs[0,1])**2)**0.5<d_bosque_TC or ((y-coord_TCs[1,0])**2+(x-coord_TCs[1,1])**2)**0.5<d_bosque_TC:
                bosques.append([y,x])
        terreno[y,x]=diccionario['terrain_id']
        # elevacion[x,y]=diccionario['elevation']
    
    coord_bosques=find_cluster_center(bosques, 6)
    
    if plot_forests==True:
        plt.figure(figsize=(8,8))
        plt.imshow(terreno,extent=axis, interpolation='none', aspect='equal')
        plt.scatter(coord_bosques[:, 1], 120-coord_bosques[:, 0], c='green', s=100, alpha=0.5)
