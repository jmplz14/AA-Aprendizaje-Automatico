#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:23:46 2019

@author: jose
"""
from sklearn.feature_selection import VarianceThreshold 
import numpy as np


def normalizarDatos(datos):
    normalizada = datos / 16
    return normalizada

def eliminarVarianza(train_X,test_X,limite = 0):
    row_train = np.size(train_X,0)
    
    datos = np.concatenate((train_X, test_X), axis=0)
    
    selector = VarianceThreshold(limite)
    datos_procesados = selector.fit_transform(datos)
    print(selector.get_support())
    train = datos_procesados[:row_train, :]
    test = datos_procesados[row_train:, :]
    return train,test


def loadCSV(fichero):
    
    my_data = np.genfromtxt(fichero, delimiter=',')
    clases = my_data[:, -1] 
    datos = my_data[:, :-1]
    #datos = normalizarDatos(datos)
    
    return datos,clases




def main():
    train_X, train_y = loadCSV("datos/optdigits.tra")
    test_X, test_y = loadCSV("datos/optdigits.tes")
    
    train_X = normalizarDatos(train_X)   
    test_X = normalizarDatos(test_X)
    
    train_X,test_y = eliminarVarianza(train_X,test_X, 0.001);
    print(np.size(train_X,1))
    
    
    
 
    
if __name__== "__main__":
  main()
  
  
