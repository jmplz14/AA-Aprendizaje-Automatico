#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:20:22 2019

@author: jose
"""
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
def obtenerValorPesos(punto, pesos):
    return [pesos[0] + pesos[1]*punto[0] + pesos[2]*punto[0], pesos[1] + pesos[1]*punto[1] + pesos[2]*punto[1]]

def dibujarGrafico2D(X, Y, clases, pesos, numero, labelx, labely, titulo):
    plt.figure(numero)
    x = np.linspace(0,0.6,100)
    plt.plot(x,-pesos[0]/pesos[2] - pesos[1]/pesos[2]*x, 'r-') 
    plt.title(titulo)
    plt.scatter(X,Y, c=clases)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    cinco_patch = mpatches.Patch(color='yellow', label='5')
    uno_patch = mpatches.Patch(color='purple', label='1')

    plt.legend(handles=[cinco_patch,uno_patch])
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

def h(x, pesos):
    solucion = np.zeros(len(x),np.float64)
    i = 0
    for i in range(len(solucion)):
        solucion[i] = np.dot(np.transpose(pesos),x[i])
        
    return solucion
   
    
    
    
def gradienteDescendenteEstocastico(x, y, x_test, y_test, num_iteraciones,tam_minibatch,tasa_aprendizaje):
    pesos = np.zeros(x[0].size)
    tamano_muestra = y.size
    print(y.size)
    permutacion = np.random.permutation(tamano_muestra)
    pos_permutacion = 0
    pos_minibatch = 0
    iteraciones = 0
    mejor_error = 10000
    mejor_pesos = None
    x_minibatch = np.empty((tam_minibatch,3))
    y_minibatch = np.empty(tam_minibatch)
    while iteraciones < num_iteraciones:
        
        if pos_minibatch == tam_minibatch:
            suma=np.zeros(len(x[1]), np.float64)
            suma += np.dot(np.transpose(x_minibatch),( h(x_minibatch,pesos)-y_minibatch ))
            pesos=pesos-tasa_aprendizaje*(2.0/tam_minibatch)*suma
            iteraciones += 1
            error = error_out(x_test,y_test,pesos)
            if mejor_error > error:
                mejor_pesos = pesos
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
        
    return mejor_pesos
       
def error(x,y,pesos):
    return (1/y.size)*(pesos.T @ x.T @ x @ pesos - 2*pesos.T @ x.T @ y + y.T @ y )

"""def error_out(x,y,pesos):
    num_errores = 0
    for i in range(y.size):
        resultado = np.dot(x[i],pesos)
        if(resultado < 0 and y[i] == 1): 
            num_errores += 1
        if(resultado > 0 and y[i] == -1):
            num_errores += 1
    return 1.0*num_errores/y.size"""
    
    
    
    

def main():
    training, clases_training = obtenerDatos("datos/X_train.npy", "datos/y_train.npy")
    test, clases_test = obtenerDatos("datos/X_test.npy", "datos/y_test.npy")
    
    
    gde = gradienteDescendenteEstocastico(training, clases_training,test, clases_test, 100,128,0.1)
    Inversa = pseudoInversa(training, clases_training)
    print("Ejercicios 1")
    print("El vector de pesos dado por la pseudoinversa es: ", Inversa)
    print("El vector de pesos dado por el gradiente descendente estocasito es: ", gde)
    print("El error en el text de la pseudoinversa es: ", error(test, clases_test, Inversa))
    print("El error en el text del gradiente descendente estocastico es: ", error(test, clases_test, gde))
    print("El error en el training de la pseudoinversa es: ", error(training, clases_training, Inversa))
    print("El error en el training del gradiente descendente estocastico es: ", error(training, clases_training, gde))
    
    
    #dibujarGrafico2D(training[: , 1], training[: , 2], clases_training, Inversa)
    

    
    dibujarGrafico2D(training[: , 1], training[: , 2], clases_training, gde, 1, "Intensidad promedio", "Simetria", "Gradiente Descendente Estocastico")
    dibujarGrafico2D(training[: , 1], training[: , 2], clases_training, Inversa, 2, "Intensidad promedio", "Simetria", "Pseudoinversa")
    

if __name__== "__main__":
  main()
    
    

