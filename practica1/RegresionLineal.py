#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:20:22 2019

@author: jose
"""

import random
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt

"""def obtenerValorPesos(punto, pesos):
    return [pesos[0] + pesos[1]*punto[0] + pesos[2]*punto[0], pesos[1] + pesos[1]*punto[1] + pesos[2]*punto[1]]
"""

"""
Funcion para dibujar la grafica de la regresion lineal
""" 
def dibujarGrafico2D(X, Y, clases, pesos, numero, labelx, labely, titulo ):
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

#Funcion para dibujar la grafica de el ejercicio2c
def dibujarGraficoEjer2c(X, Y, clases, pesos, numero, labelx, labely, titulo ):
    plt.figure(numero)
    x = np.linspace(-1,1,100)
    plt.plot(x,-pesos[0]/pesos[2] - pesos[1]/pesos[2]*x, 'r-') 
    plt.title(titulo)
    plt.scatter(X,Y, c=clases)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.ylim(-1, 1)
    cinco_patch = mpatches.Patch(color='yellow', label='1')
    uno_patch = mpatches.Patch(color='purple', label='-1')

    plt.legend(handles=[cinco_patch,uno_patch])
    plt.show()
    
#funcion para cargar los datos dese los ficheros
def obtenerDatos(ficheroTraining, ficheroTest):
    
    #carga los datos de los ficheros pasdos por parmetro
    datos_training = np.load(ficheroTraining)
    clase_training = np.load(ficheroTest)
    
    #creo un enumrate con el vector de clases
    clase = enumerate(clase_training)
    
    #seleciono solo los datos que tienen 1 o 5 en su etiqueta
    clase_1y5 = [x for x in clase if x[1] == 1 or x[1] == 5]
    
    #creo la matriz para almacenar los datos de los numeros 1 y 5 antes seleccionados
    num_filas = len(clase_1y5)
    matrix_training = np.empty((num_filas,3))
    
    #sera en la matriz donde almacnaremos el -1 si es 1 o 1 si es 5
    resultados_training = np.empty(num_filas)
    
    #nos quedamos con los datos de las etiquetas que son 1 o 5
    for i in range(0,num_filas):
        matrix_training[i][0] = 1
        matrix_training[i][1] = datos_training[clase_1y5[i][0]][0]
        matrix_training[i][2] = datos_training[clase_1y5[i][0]][1]
        
        #le ponemos la nueva etiqueta
        if clase_1y5[i][1] == 1:
            resultados_training[i] = -1
        else:
            resultados_training[i] = 1
            
    return matrix_training, resultados_training

#Calculamos la pseudoinversa
def pseudoInversa(x,y):
    return np.linalg.inv(x.T @ x) @ x.T @ y 

#nos da la multiplicacion de los pesos transpuestos por la matriz x
def h(x, pesos):
    solucion = np.zeros(len(x),np.float64)
    i = 0
    for i in range(len(solucion)):
        solucion[i] = np.dot(np.transpose(pesos),x[i])
        
    return solucion
   
#Calcula el error de para sacar el ein o el eout
def obtenerError(x,y,pesos):
    return (1/y.size)*(pesos.T @ x.T @ x @ pesos - 2*pesos.T @ x.T @ y + y.T @ y )

    
    
"""
    algoritmos del gradiente descendiente estocastico
    se le pasa los datos de training y los datos de test porque yo evaluo cada pesos dados
    para ver si son los mejores y en ese caso almacenarlos.Tambien se le pasa el numero
    de iteraciones, la tasa de aprendizaje y el tam_minibatch
"""

def gradienteDescendenteEstocastico(x, y, num_iteraciones,tam_minibatch,tasa_aprendizaje):
    #inicio el vector de pesos
    pesos = np.zeros(x[0].size)
    
    """
    obtengo el tamaño de la muestra para sacar una permutacion aleatoria de
    los datos
    """
    tamano_muestra = y.size
    permutacion = np.random.permutation(tamano_muestra)
    #inicializo a cero las posciones de lectura del la permutacion y de los minibatch
    pos_permutacion = 0
    pos_minibatch = 0
    
    #inicializo el valor de iteraciones que llevamos a 0
    iteraciones = 0
    
    #pongo un mejor_error muy alto para que pueda mejorarse en la primera pasada
    #mejor_error = 10000
    #los mejores pesos los almaceno como None al principio
    #mejor_pesos = None
    
    #inicio los minibatch a vacio
    x_minibatch = np.empty((tam_minibatch,3))
    y_minibatch = np.empty(tam_minibatch)
    
    #reocrro hasata llegar al maximo de iteraciones
    while iteraciones < num_iteraciones:
        """
        si llevamos la posicion de minibacht igual al tamaño d eestos ya podemos
        calcular los pesos para esta pasada
        """
        if pos_minibatch == tam_minibatch:
            #Calculo los pesos
            suma=np.zeros(len(x[1]), np.float64)
            suma += np.dot(np.transpose(x_minibatch),( h(x_minibatch,pesos)-y_minibatch ))
            pesos=pesos-tasa_aprendizaje*(2.0/tam_minibatch)*suma
            
            """ #obtengo el error
            error = obtenerError(x,y,pesos)
            #y lo comparao para ver si es el mor
            if mejor_error > error:
                mejor_pesos = pesos
            """
            #limpio los minitbatch
            x_minibatch = np.empty((tam_minibatch,3))
            y_minibatch = np.empty(tam_minibatch)
            #pesos = calcularPesos(x_minibatch, pesos);
            
            #vuelvo la posicion del minibatch a 0
            pos_minibatch = 0
        """
        si la posicion de permutacion es el final del vector es que hemos completado
        una iteracion y hay que crear una nueva permutacion inicializando a cero los datos
        """
        if pos_permutacion == tamano_muestra:
            pos_permutacion = 0
            pos_minibatch = 0
            #cuando esto se da se tiene que hemos dado una iteracion
            iteraciones += 1
            permutacion=np.random.permutation(tamano_muestra)
            x_minibatch = np.empty((tam_minibatch,3))
            y_minibatch = np.empty(tam_minibatch)
            
        
        #vamos almacenando las minibatch para ser calculado luego con ellos
        x_minibatch[pos_minibatch][0] = x[permutacion[pos_permutacion]][0]
        x_minibatch[pos_minibatch][1] = x[permutacion[pos_permutacion]][1]
        x_minibatch[pos_minibatch][2] = x[permutacion[pos_permutacion]][2]
        y_minibatch[pos_minibatch] = y[permutacion[pos_permutacion]]
        
        #se suma una posicion a la permutacion y a el minibatch
        pos_permutacion += 1
        pos_minibatch += 1
    
    #devolvemos los mejores pesos    
    return pesos
       


def simula_unif(num_coordenadas, dimension, rango):
    matriz = np.random.uniform(low=rango[0], high=rango[1], size=(num_coordenadas, dimension))
    return matriz

#Dibujamos las matrices de puntos
def dibujar_matriz(matriz, numero, x, y, titulo):
    plt.figure(numero)
    plt.plot(matriz[: , 0], matriz[:, 1], 'ko', c = "blue")
    
    plt.show()

#dibujamos los graficos del ejercicio 2
def dibujarGraficoEjer2(X, Y, clases, numero, labelx, labely, titulo):
    plt.figure(numero)
    #x = np.linspace(0,0.6,100)
    #plt.plot(x,-pesos[0]/pesos[2] - pesos[1]/pesos[2]*x, 'r-') 
    plt.title(titulo)
    plt.scatter(X,Y, c=clases)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    cinco_patch = mpatches.Patch(color='yellow', label='1')
    uno_patch = mpatches.Patch(color='purple', label='-1')

    plt.legend(handles=[cinco_patch,uno_patch])
    plt.show()
    
#añadimos las etiquetas a los datos con la funcion dada por el ejercicio
def formula2b(matriz): 
    #definimos la formula
    formula = lambda th: np.sign((th[0] - 0.2)**2 + th[1]**2 -0.6)
    
    #sacamos el tamaño de la matriz
    tamano = int(matriz.size / 2) 
    #inicalizamos la matriz que devolveremos
    resultado = np.empty((tamano,4), np.float64)
    
    for i in range(0,tamano):
        #añadimos la columna de 1
        resultado[i][0] = 1
        #los valores de x e y
        resultado[i][1] = matriz[i][0]
        resultado[i][2] = matriz[i][1]
        #la nueva etiqueta
        resultado[i][3] = formula([matriz[i][0], matriz[i][1]])

    #insertamos el ruido para el 10% de las muestras
    num_cambios = int(tamano * 0.1)
    posiciones_cambiar = random.sample(range(tamano), num_cambios)
    for i in posiciones_cambiar:
 
        if(resultado[i][3] == 1):
            resultado[i][3] = -1
        else:
            resultado[i][3] = 1
   # for i in (0,num_cambios):
        
    
    return resultado
    

def main():
    
    #cargamos los datos 
    training, clases_training = obtenerDatos("datos/X_train.npy", "datos/y_train.npy")
    test, clases_test = obtenerDatos("datos/X_test.npy", "datos/y_test.npy")
    
    #calculamos el gradiente descendente estocastico
    gde = gradienteDescendenteEstocastico(training, clases_training, 100,128,0.01)
    #calculamos la pseudoinversa
    Inversa = pseudoInversa(training, clases_training)
    print("\n..............................................................")
    print("Ejercicios 1")
    print("El vector de pesos dado por la pseudoinversa es: ", Inversa)
    print("El vector de pesos dado por el gradiente descendente estocasito es: ", gde)
    print("El error en el text de la pseudoinversa es: ", obtenerError(test, clases_test, Inversa))
    print("El error en el text del gradiente descendente estocastico es: ", obtenerError(test, clases_test, gde))
    print("El error en el training de la pseudoinversa es: ", obtenerError(training, clases_training, Inversa,))
    print("El error en el training del gradiente descendente estocastico es: ", obtenerError(training, clases_training, gde))
    
    
    #dibujarGrafico2D(training[: , 1], training[: , 2], clases_training, Inversa)
    

    #Dibujamos los grficos para la pseudoinversa y el gde
    dibujarGrafico2D(training[: , 1], training[: , 2], clases_training, gde, 1, "Intensidad promedio", "Simetria", "Gradiente Descendente Estocastico")
    dibujarGrafico2D(training[: , 1], training[: , 2], clases_training, Inversa, 2, "Intensidad promedio", "Simetria", "Pseudoinversa")
    
    #Calculaos la matriz con los putnos uniformes 
    matriz_uniforme = simula_unif(1000, 2, (-1,1))
    
    
    
    #añadimos la etiquetas 1 y -1 
    matriz_2b = formula2b(matriz_uniforme)
    
    
    #calculamos los pesos para el gradiente descendente estocastico de los datos anteriores
    gde_ejer2c = gradienteDescendenteEstocastico(matriz_2b[:, 0:3], matriz_2b[: , 3].T, 100,128,0.01)
    #dibujarGraficoEjer2c(matriz_2b[: , 1], matriz_2b[: , 2], matriz_2b[: , 3].T, gde_ejer2c, 5, "x", "y", "Ejercicio 2 c" )
    
    print("\n..............................................................")
    print("Ejercicio 2 a")
    #dibujamos la matriz
    dibujar_matriz(matriz_uniforme,3,"x","y", "matriz puntos")
    
    print("Ejercicio 2 b")
    dibujarGraficoEjer2(matriz_2b[: , 1], matriz_2b[: , 2], matriz_2b[: , 3].T, 4, "x", "y", "Ejercicio 2 b")
    
    print("Grafica del ejercicio 2 c")
    dibujarGraficoEjer2c(matriz_2b[: , 1], matriz_2b[: , 2], matriz_2b[: , 3].T, gde_ejer2c, 5, "x", "y", "Ejercicio 2 c" )
    print("Pesos del ejercicio 2 c", gde_ejer2c)
    print("El error Ein gradiente descendente estocastico es: ", obtenerError(matriz_2b[:, 0:3], matriz_2b[: , 3].T, gde_ejer2c))
   
    #iniciamos los errores para luego hacer la media a 0
    Ein_acumulados = 0
    Eout_acumulados = 0
    
    #realizamos 1000 iteraciones 
    for i in range(1000):
        #generamos tanto los datos de test como los de training
        training_2d = simula_unif(1000, 2, (-1,1))
        test_2d = simula_unif(1000, 2, (-1,1))
        
        #Le metemos el ruido y los etquetamos
        training_modificado = formula2b(training_2d)
        test_modificado = formula2b(test_2d)
        
        #sacamos lso pesos con el gradiente descendiente estocastico 
        pesos = gradienteDescendenteEstocastico(training_modificado[:, 0:3], training_modificado[: , 3].T, 50,128,0.01)
        
        #acumulamos los errores para calcular la media
        Ein_acumulados += obtenerError(training_modificado[:, 0:3], training_modificado[: , 3].T, pesos)
        Eout_acumulados += obtenerError(test_modificado[:, 0:3], test_modificado[: , 3].T, pesos)
    
    print("\n..............................................................")
    print("Ejericico 2 d")
    print("El error medio Ein es: ", Ein_acumulados/1000)
    print("El error medio Eout es: ", Eout_acumulados/1000)
    
       
        

if __name__== "__main__":
  main()
    
    

