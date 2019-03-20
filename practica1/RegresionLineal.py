#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:20:22 2019

@author: jose
"""
import numpy as np

def obtenerDatos(ficheroTraining, ficheroTest):

    datos_training = np.load(ficheroTraining)
    clase_training = np.load(ficheroTest)
    clase = enumerate(clase_training)
    
    clase_1y5 = [x for x in clase if x[1] == 1 or x[1] == 5]
    
    
    num_filas = len(clase_1y5)
    matrix_training = np.empty((num_filas,3))
    
    resultados_training = np.empty(num_filas)
    
    for i in range(0,num_filas):
        matrix_training[i][0] = 1
        matrix_training[i][1] = datos_training[clase_1y5[i][0]][0]
        matrix_training[i][2] = datos_training[clase_1y5[i][0]][1]
        
        
        if clase_1y5[i][1] == 1:
            resultados_training[i] = -1
        else:
            resultados_training[i] = 1
    return matrix_training, resultados_training


def pseudoInversa(x,y):
    return np.linalg.inv(x.T @ x) @ x.T @ y

def main():
    training, clases_training = obtenerDatos("datos/X_train.npy", "datos/y_train.npy")
    test, clases_test = obtenerDatos("datos/X_test.npy", "datos/y_test.npy")
    
    
    Inversa = pseudoInversa(training, clases_training)
    
    print(Inversa)
    
    
    

if __name__== "__main__":
  main()
    
    

