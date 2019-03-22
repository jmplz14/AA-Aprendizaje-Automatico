#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:20:22 2019

@author: jose
"""
import numpy as np
import matplotlib.pyplot as plt
def obtenerValorPesos(punto, pesos):
    return [pesos[0] + pesos[1]*punto[0] + pesos[2]*punto[0], pesos[1] + pesos[1]*punto[1] + pesos[2]*punto[1]]

def dibujarGrafico2D(X, Y, clases, pesos):
    x = np.linspace(0,0.6,100)
    plt.plot(x,-pesos[0]/pesos[2] - pesos[1]/pesos[2]*x, 'r-') 
    plt.scatter(X,Y, c=clases)
    plt.show()
    

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

def h(x_, w_):
    sol = np.zeros(len(x_),np.float64)
    i = 0
    for i in range(len(sol)):
        sol[i] = np.dot(np.transpose(w_),x_[i])
        
    return sol
   
    
    
    
def gradienteDescendenteEstocastico(x, y,num_iteraciones,tam_minibatch,tasa_aprendizaje):
    pesos = np.zeros(x[0].size)
    tamano_muestra = y.size
    print(y.size)
    permutacion = np.random.permutation(tamano_muestra)
    pos_permutacion = 0
    pos_minibatch = 0
    iteraciones = 0
    
    x_minibatch = np.empty((tam_minibatch,3))
    y_minibatch = np.empty(tam_minibatch)
    while iteraciones < num_iteraciones:
        
        if pos_minibatch == tam_minibatch:
            suma=np.zeros(len(x[1]), np.float64)
            suma += np.dot(np.transpose(x_minibatch),( h(x_minibatch,pesos)-y_minibatch ))
            pesos=pesos-tasa_aprendizaje*(2.0/tam_minibatch)*suma
            iteraciones += 1
            x_minibatch = np.empty((tam_minibatch,3))
            y_minibatch = np.empty(tam_minibatch)
            #pesos = calcularPesos(x_minibatch, pesos);
            
            pos_minibatch = 0
            
        if pos_permutacion == tamano_muestra:
            pos_permutacion = 0
            pos_minibatch = 0
            permutacion=np.random.permutation(tamano_muestra)
            x_minibatch = np.empty((tam_minibatch,3))
            y_minibatch = np.empty(tam_minibatch)
            
        
        
        x_minibatch[pos_minibatch][0] = x[permutacion[pos_permutacion]][0]
        x_minibatch[pos_minibatch][1] = x[permutacion[pos_permutacion]][1]
        x_minibatch[pos_minibatch][2] = x[permutacion[pos_permutacion]][2]
        y_minibatch[pos_minibatch] = y[permutacion[pos_permutacion]]
        
        pos_permutacion += 1
        pos_minibatch += 1
        
    return pesos
        
    
        
    
    
    
    
    
    

def main():
    training, clases_training = obtenerDatos("datos/X_train.npy", "datos/y_train.npy")
    test, clases_test = obtenerDatos("datos/X_test.npy", "datos/y_test.npy")
    
    
    
    Inversa = pseudoInversa(training, clases_training)
    
    
    #dibujarGrafico2D(training[: , 1], training[: , 2], clases_training, Inversa)
    
    pesos = gradienteDescendenteEstocastico(training, clases_training,1000,64,0.01)
    
    dibujarGrafico2D(training[: , 1], training[: , 2], clases_training, pesos)

if __name__== "__main__":
  main()
    
    

