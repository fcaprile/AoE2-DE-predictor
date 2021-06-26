# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 19:11:48 2021

@author: ferchi
"""


from sklearn.cluster import KMeans

def find_cluster_center(X,n_cluster):
    cluster_finder=KMeans(n_cluster)
    cluster_finder.fit(X)
    return cluster_finder.cluster_centers_

    