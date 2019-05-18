#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:23:46 2019

@author: jose
"""
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Perceptron
from time import time
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import confusion_matrix

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


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

def regresionLinealEntrenarDefecto(X,y,X_test, y_test):
    
    model = LogisticRegression( )
    #model = LogisticRegression(penalty = None,solver = 'newton-cg')
    
    model.fit(X,y)
    
    tasa_acierto = model.score(X_test,y_test)
    return tasa_acierto

def regresionLinealEntrenarMejorado(X,y,X_test, y_test,regurlarizacion = 'l2' , ajuste = 1):
    
    model = LogisticRegression(solver = 'liblinear', multi_class='ovr', penalty = 'l1', C = ajuste, random_state=0 )
    #model = LogisticRegression(penalty = None,solver = 'newton-cg')
    
    model.fit(X,y)
    
    tasa_acierto = model.score(X_test,y_test)
    return tasa_acierto,model.predict(X_test)
    
def anadirInformacionPolinomial(train_X, test_X, grado = 2):
    poly = PolynomialFeatures(grado)
    train_X = poly.fit_transform(train_X)
    test_X = poly.fit_transform(test_X)
    
    return train_X, test_X
def perceptronEntrenarMejorado(X,y,X_test, y_test):
    clf = Perceptron(tol=1e-3, max_iter = 1000, n_iter_no_change = 5, random_state=0, penalty = 'l1', alpha=0.000001)

    clf.fit(X, y)  
    clf.score(X, y)
    acierto = clf.score(X_test, y_test)
    return acierto;

def perceptronEntrenarDefecto(X,y,X_test, y_test):
    clf = Perceptron()

    clf.fit(X, y)  
    clf.score(X, y)
    acierto = clf.score(X_test, y_test)
    return acierto;

"""def obtenerDatosRegularizacion(train_X,train_y,test_X,test_y, regularizacion):
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
        print(i*2,",",acierto)"""
def plot_confusion_matrix(df_confusion, title='Matriz de confusion', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    plt.title(title)
    plt.colorbar()
    #tick_marks = np.arange(len(df_confusion.columns))
    #plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    #plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    #plt.ylabel(df_confusion.index.name)
    #plt.xlabel(df_confusion.columns.name)




def main():
    train_X, train_y = loadCSV("datos/optdigits.tra")
    test_X, test_y = loadCSV("datos/optdigits.tes")
    
    train_X = normalizarDatos(train_X)   
    test_X = normalizarDatos(test_X)
    
    train_X,test_X = eliminarVarianza(train_X,test_X, 0);
    
    train_X,test_X = anadirInformacionPolinomial(train_X,test_X,2)
    regularizacion = 'l1'
    


    start_time = time()
    acierto_logistico = regresionLinealEntrenarDefecto(train_X,train_y,test_X,test_y)
    print("Regresion logistica por defecto")
    print("Tiempo=", time() - start_time )
    print("Eout=",(1-acierto_logistico)*100)
    
    
    start_time = time()
    acierto_perceptron = perceptronEntrenarDefecto(train_X,train_y,test_X,test_y)
    print("Percetron por defecto")
    print("Tiempo=", time() - start_time )
    print("Eout=",(1-acierto_perceptron)*100)
    
    start_time = time()
    acierto_logistico,y_predecido = regresionLinealEntrenarMejorado(train_X,train_y,test_X,test_y,regularizacion,500)
    print("Regresion logistica mejorada")
    print("Tiempo=", time() - start_time )
    print("Eout=",(1-acierto_logistico)*100)
    
    
    
    matriz_confusion = confusion_matrix(test_y, y_predecido)
    matriz_confusion =  matriz_confusion /  matriz_confusion.sum(axis=1) 
    
    plot_confusion_matrix(matriz_confusion)
    
    
    
    
    
    
    
    
    
    #obtenerDatosRegularizacion(train_X,train_y,test_X,test_y, regularizacion)
    """l1 = np.genfromtxt("datos/l1.csv", delimiter=',')
    l2 = np.genfromtxt("datos/l2.csv", delimiter=',')
    
    p = np.polyfit(l1[:,0],l1[:,1], 1)
    p2 = np.polyfit(l2[:,0],l2[:,1], 1)
    
    # Valores de y calculados del ajuste
    y_ajuste = p[0]*l1[:,0] + p[1]
    y_ajuste2 =p2[0]*l2[:,0] + p2[1]
    # Dibujamos los datos experimentales
    p_datos, = plt.plot(l1[:,0], l1[:,1], 'r.')
    # Dibujamos la recta de ajuste

    #plt.plot(l1[:,0], y_ajuste, 'g-')
    plt.plot(l1[:,0], y_ajuste2, 'b-')
    plt.title('Mejora con metrica l1')
    
    plt.xlabel('Parametro C')
    plt.ylabel('Tasa Acierto')
    plt.axis([2,500,0.9750,0.9825])
    blue_patch = mpatches.Patch(color='red', label='Medidas tomadas')
    green_patch = mpatches.Patch(color='blue', label='Funcion ajustada')
    
    #añadimos al legend los distintos tipos
    plt.legend(handles=[blue_patch,green_patch])
    
    plt.show()
    #Calculo de mejoras con regularizacion l1 y l2.
    #obtenerDatosRegularizacion(train_X,train_y,test_X,test_y, regularizacion)"""
if __name__== "__main__":
  main()
  
  
