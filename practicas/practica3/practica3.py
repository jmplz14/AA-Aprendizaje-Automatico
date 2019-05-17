#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:23:46 2019

@author: jose
"""
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold 
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


def normalizarDatos(datos):
    normalizada = datos / 16
    return normalizada

def eliminarVarianza(train_X,test_X,limite = 0):
    row_train = np.size(train_X,0)
    
    datos = np.concatenate((train_X, test_X), axis=0)
    
    selector = VarianceThreshold(limite)
    datos_procesados = selector.fit_transform(datos)

    train = datos_procesados[:row_train, :]
    test = datos_procesados[row_train:, :]
    return train,test


def loadCSV(fichero):
    
    my_data = np.genfromtxt(fichero, delimiter=',')
    clases = my_data[:, -1] 
    datos = my_data[:, :-1]
    #datos = normalizarDatos(datos)
    
    return datos,clases

def regresionLinealEntrenar(X,y,X_test, y_test,regurlarizacion = 'l2' , ajuste = 1):
    
    model = LogisticRegression(solver = 'liblinear', multi_class='ovr', penalty = 'l1', C = ajuste )
    model.fit(X,y)

    #predictions = model.predict(X)
    #print(model.score(X,y))
    tasa_acierto = model.score(X_test,y_test)
    return tasa_acierto
    
def anadirInformacionPolinomial(train_X, test_X, grado = 2):
    poly = PolynomialFeatures(grado)
    train_X = poly.fit_transform(train_X)
    test_X = poly.fit_transform(test_X)
    
    return train_X, test_X

def obtenerDatosRegularizacion(train_X,train_y,test_X,test_y, regularizacion):
    datos = np.empty((250,2))
    for i in range(1,251):
        acierto = 0
        if regularizacion == 'l1':
            for j in range(0,5):
                acierto += regresionLinealEntrenar(train_X,train_y,test_X,test_y,regularizacion,i*2)
            acierto = acierto/5;
        else:
            acierto = regresionLinealEntrenar(train_X,train_y,test_X,test_y,regularizacion,i*2)
            
        datos[i-1][0] = i*2
        datos[i-1][1] = acierto
        print(i*2,",",acierto)
    if regularizacion == 'l1':
        np.savetxt("datos/l1.csv", datos, delimiter=",")
    else:
        np.savetxt("datos/l2.csv", datos, delimiter=",")


def main():
    train_X, train_y = loadCSV("datos/optdigits.tra")
    test_X, test_y = loadCSV("datos/optdigits.tes")
    
    train_X = normalizarDatos(train_X)   
    test_X = normalizarDatos(test_X)
    
    train_X,test_X = eliminarVarianza(train_X,test_X, 0);
    
    train_X,test_X = anadirInformacionPolinomial(train_X,test_X,2)
    regularizacion = 'l1'
    
    
    obtenerDatosRegularizacion(train_X,train_y,test_X,test_y, regularizacion)
    
if __name__== "__main__":
  main()
  
  
