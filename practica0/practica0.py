# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
import numpy as np
import math 
import matplotlib.pyplot as plt
from sklearn import datasets as ds
import matplotlib.patches as mpatches
iris = ds.load_iris()
#coge todas las columnas apartir de la de la segunda
x = iris.data[:, 2:]  
y = iris.target

'''
almaceno los elementos que tienen nuestra muestra -1 para que al relizar el 
el range podamos acceder a todos los datos de la muestra para dibujarla    
'''
n = y.size - 1

#dibujamos el punto cambiando el color para cada tipo de clase
for i in range(0, n):
    if y[i] == 0:
        plt.scatter(x[i][0], x[i][1], marker='o', color='blue')
    elif y[i] == 1:
        plt.scatter(x[i][0], x[i][1], marker='o', color = 'green')
    else:
        plt.scatter(x[i][0], x[i][1], marker='o', color = 'red')
        
#ponemos nombres a los label para x e y
plt.xlabel('x')
plt.ylabel('y')

#creamos la cadena para cada tipo del legend
blue_patch = mpatches.Patch(color='blue', label='0')
green_patch = mpatches.Patch(color='green', label='1')
red_patch = mpatches.Patch(color='red', label='2')

#añadimos al legend los distintos tipos
plt.legend(handles=[blue_patch,green_patch,red_patch])

#le ponemos el nombre al grafico
plt.title('flores')

#lo dibujamos por pantalla 
plt.show()

print('\npulsa tecla enter para pasar a ver las proporciones de nuestros conjuntos para el ejercicio 2')
input()



'''
Al hacer enumerate con el vecotr y tendremos la (numero , clase) donde numero
identifica la posicion que tendria en el vector de datos y la calase seria el tipo
al que pertenece
'''

flores = list(enumerate(y))
#print(list(zip(x,y)))


#separamos las flores de cada clase en una lista
clase0 =[x for x in flores if x[1] == 0]
clase1 =[x for x in flores if x[1] == 1]
clase2 =[x for x in flores if x[1] == 2]


'''
en en los vectores tendremos una lista de numeros aleatorios que tendra el rango
maximo el numero de elementos que tenga la clase -1 y se generara un tercio de
lo que corresponde a cada clase para que la muestra tenga los mismas proporciones
que la muestra inicial
'''
valores0 = list(np.random.choice(len(clase0), int((0.2*(n+1))/3), replace=False))
valores1 = list(np.random.choice(len(clase1), int((0.2*(n+1))/3), replace=False))
valores2 = list(np.random.choice(len(clase2), int((0.2*(n+1))/3), replace=False))
'''print(valores0)
print(valores1)
print(valores2)'''

test = list()
'''
creamos el array test con ctraining0 = 0ada flor de la clase seleccionada aleatoriamente y de 
la que guardamos la posicion en los vectores valores
'''
for i in range(0,len(valores0)):
    test.append(clase0[valores0[i]])
    test.append(clase1[valores1[i]])
    test.append(clase2[valores2[i]])
    
'''
en training nos quedamos con las flores que no esten en test y con esto tambien
nos asegurames que en trainin tendremos el 80% de los datos y con la mismas 
proporciones al incluir las flores que no estan en test
'''
training = [x for x in flores if x not in test]

#las ordeno solamente para ver que coninciden y no hay repetidas entre los dos
test.sort()
training.sort()

'''
mostramos el tamaño de los vectores para ver que coninciden con las proporciones
correctas
'''
print('tamaño del vector del training es ' , len(training))
print('tamaño del vector del test es ' , len(test))

training0 = 0
training1 = 0
training2 = 0
test0 = 0
test1 = 0
test2 = 0

'''
estos dos for solo nos cuenta el numero de variables de cada tipo que hay en 
training y en test
'''
for i in training:
    if i[1] == 0:
        training0 += 1
    elif i[1] == 1:
        training1 += 1
    else:
        training2 += 1
        
for i in test:
    if i[1] == 0:
        test0 += 1
    elif i[1] == 1:
        test1 += 1
    else:
        test2 += 1 
        
print('En training tenemos ', training0 , ' de la clase 0')
print('En training tenemos ', training1 , ' de la clase 1')
print('En training tenemos ', training2 , ' de la clase 2')

print('En test tenemos ', test0 , ' de la clase 0')
print('En test tenemos ', test1 , ' de la clase 1')
print('En test tenemos ', test2 , ' de la clase 2')

print('\npulsa tecla enter para pasar a ver los datos de test y training')
input()

'''
mostramos los conjuntos de test y training
'''
print('''\nEl formato de nuestro conjunto de test y training nos da dos numeros 
por cada elemento de la lista. El segundo numero corresponde la clase a la
que pertenece y el primero es la posicion que tendria en la variable x 
donde almaceno las caracteristicas de la flor.
De forma que usando el primer numero podriamos haceder a las caracteristicas
de la flor en la variable x. He usado este formato simplemente porque 
es mas sencillo ver que he realizado bien la separacion y que no hay flores
repetidas con un simple vistazo a los datos. 
Se accederian a las caracteristicas de las flores de la siguiente forma\n''')

print('x[test[0][1]] se accederia a las caracteristicas de la primera flor de nuestro datos de test\n')

#print(x[test[0][1]])
print('---test---')
print(test)
print('---training---')
print(training)

print('\npulsa tecla enter para pasar al ejercicio 3 y ver 100 numeros equidistantes entre 0 y 2*pi')
input()
#ejercicio 3

#genero los 100 numeros
numeros = np.linspace(0,2*math.pi,100)
print(numeros)

#creo las listas donde almacenare los valores de las funciones
sin = list()
cos = list()
cossin = list()

#Se van añadiendo los valores a cada lista correspondiente
for n in numeros:
    sin.append(math.sin(n))
    cossin.append(math.sin(n)+math.cos(n))
    cos.append(math.cos(n))
    

print('\npulsa tecla enter para ver el resultado de sin')
input()
print(sin)

print('\npulsa tecla enter para ver el resultado de cos')
input()
print(cos)

print('\npulsa tecla enter para ver el resultado de sin + cos')
input()
print(cossin)

print('\npulsa tecla enter para ver la grafica de las tres funciones')
input()
plt.plot(numeros,sin, '--', color = 'black')
plt.plot(numeros,cos, '--',color = 'blue')
plt.plot(numeros,cossin, '--',color = 'red')

plt.show()


    

