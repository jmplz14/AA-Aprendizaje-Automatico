#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 18:23:48 2019

@author: jose
"""
import matplotlib.patches as mpatches
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from sympy.abc import k,n
from sympy.utilities.lambdify import lambdify, implemented_function


    

"""def gradiante2(inicio: list, tasaAprendizaje: np.float64(), formula, u, v,num_iteraciones, minimo):

    #formula=formula.subs({u:inicio[0], v: inicio[1]})
    #formula = formula.diff(u)
    datos = list()
    
    
    der_parcial_u = formula.diff(u)
    der_parcial_v = formula.diff(v)
    #gradiente = der_parcial_u + der_parcial_v
    
    datos.append([inicio[0],inicio[1],np.float64(formula.subs({u:inicio[0], v: inicio[1]}))])
    
    valor_u = inicio[0] - tasaAprendizaje * np.float64(der_parcial_u.subs({u:inicio[0], v: inicio[1]}))
    valor_v = inicio[1] - tasaAprendizaje * np.float64(der_parcial_v.subs({u:inicio[0], v: inicio[1]}))
    i = 1;
    #while np.float64(formula.subs({u:valor_u, v: valor_v})) > minimo and i < num_iteraciones:
    while i < num_iteraciones:
        
        antiguo_u = valor_u
        valor_u = valor_u - tasaAprendizaje * np.float64(der_parcial_u.subs({u:valor_u, v: valor_v}))
        valor_v = valor_v - tasaAprendizaje * np.float64(der_parcial_v.subs({u:antiguo_u, v: valor_v}))
        datos.append([valor_u,valor_v,np.float64(formula.subs({u:valor_u, v: valor_v}))])
        i = i + 1
        #print( i, ":", valor_u , ":", valor_v, np.float64(gradiente.subs({u:valor_u, v: valor_v})))
        
    #print( np.float64(formula.subs({u:valor_u, v: valor_v})))
    return datos
      
   #print(np.float64(gradiente.subs({u:valor_u, v: valor_v})))
   
   
    
    #print("(",inicio[0],",",inicio[1],")", tasaAprendizaje, "formula = " , formula)
"""  

def gradiante(der_parcial_u,der_parcial_v, u, v, tasaAprendizaje, coordenada):
    valor_u = coordenada[0] - tasaAprendizaje \
    * np.float64(der_parcial_u.subs({u:coordenada[0], v: coordenada[1]}))
    valor_v = coordenada[1] - tasaAprendizaje \
    * np.float64(der_parcial_v.subs({u:coordenada[0], v: coordenada[1]}))
    return valor_u,valor_v

def ejercicio2(inicio: list, tasaAprendizaje: np.float64(),lam_formula, formula, u, v,num_iteraciones, minimo):
    der_parcial_u = formula.diff(u)
    der_parcial_v = formula.diff(v)
    """print("dervidasas")
    print(der_parcial_u)
    print(der_parcial_v)
    print("----------")"""
    datos = np.empty((num_iteraciones,3,))
    datos[:] = np.nan
    datos[0][0] = inicio[0]
    datos[0][1] = inicio[1]
    datos[0][2] = lam_formula(inicio[0],inicio[1])
    i = 1;
    valor_u,valor_v = gradiante(der_parcial_u, der_parcial_v, u, v, tasaAprendizaje,inicio)
    print("Apartado1")
    print("Ejercicio 1")
    print("Valores para E(u,v) = (1,1) = (",valor_u,",",valor_v,")")
    while lam_formula(valor_u,valor_v) > minimo and i < num_iteraciones:
        datos[i][0] = valor_u
        datos[i][1] = valor_v
        datos[i][2] = lam_formula(valor_u,valor_v)
        
        valor_u,valor_v = gradiante(der_parcial_u, der_parcial_v, u, v, tasaAprendizaje,[valor_u,valor_v])
        
        i = i + 1
        
    if i < num_iteraciones:
        datos[i][0] = valor_u
        datos[i][1] = valor_v
        datos[i][2] = lam_formula(valor_u,valor_v)
        datos.resize(i+1,3)
        
    
    return datos

def ejercicio3(inicio: list, tasaAprendizaje: np.float64(),lam_formula, formula, u, v,num_iteraciones):
    der_parcial_u = formula.diff(u)
    der_parcial_v = formula.diff(v)
    datos = np.empty((num_iteraciones,3,))
    datos[:] = np.nan
    datos[0][0] = inicio[0]
    datos[0][1] = inicio[1]
    datos[0][2] = lam_formula(inicio[0],inicio[1])
    i = 1;
    valor_u,valor_v = gradiante(der_parcial_u, der_parcial_v, u, v, tasaAprendizaje,inicio)
    
    while i < num_iteraciones:
        datos[i][0] = valor_u
        datos[i][1] = valor_v
        datos[i][2] = lam_formula(valor_u,valor_v)
        
        valor_u,valor_v = gradiante(der_parcial_u, der_parcial_v, u, v, tasaAprendizaje,[valor_u,valor_v])
        i = i + 1
    
    return datos
    
def dibujarGrafica(datos,rango_u,rango_v,num_muestras, lam_formula,num, labelx, labely, titulo):
    

    plt.figure(num)
    generar_u = np.linspace(rango_u[0], rango_u[1], num_muestras)
    generar_v = np.linspace(rango_v[0], rango_v[1], num_muestras)
    generar_z =  np.zeros((num_muestras, num_muestras))
    for i, valor_u in enumerate(generar_u):
        for j, valor_v in enumerate(generar_v):
            generar_z[j,i] = lam_formula(valor_u,valor_v)
            
    
    tamano = int(datos.size / 3)-1
    plt.plot(datos[0][0], datos[0][1], "o", c="white")
    for i in range(1,tamano - 1):
        plt.plot(datos[i][0], datos[i][1], "o", c="red")
    
    plt.plot(datos[tamano][0], datos[tamano][1], "o", c="green")
    plt.title(titulo)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    
    blue_patch = mpatches.Patch(color='white', label='Punto Inicial')
    green_patch = mpatches.Patch(color='red', label='Puntos Intermedios')
    red_patch = mpatches.Patch(color='green', label='Punto Final')
    
    #añadimos al legend los distintos tipos
    plt.legend(handles=[blue_patch,green_patch,red_patch])
    
    
    
 
    plt.contourf(generar_u, generar_v, generar_z, num_muestras)
    plt.colorbar()
    plt.show()
   
def obtenerMenor(datos):
    x_menor = datos[0][0]
    y_menor = datos[0][1]
    z_menor = datos[0][2]
    for i in range(1,49):
        if z_menor > datos[i][2]:
            x_menor = datos[i][0]
            y_menor = datos[i][1]
            z_menor = datos[i][2]
            
    print("(", x_menor,",",y_menor, ") =", z_menor)   
            
    
def main():

    f2 = implemented_function('f', lambda k,n: np.float64((k**2 * sym.exp(n) - 2*n**2*sym.exp(-k))**2))
    lam_formula_ejer2 = lambdify((k,n), f2(k,n))
    
    u = sym.Symbol('u')
    v = sym.Symbol('v')
    formula_ejer2 = (u**2 * sym.exp(v) - 2*v**2*sym.exp(-u))**2
    #formula = u**2 + v**2
    datos_ejer2 = ejercicio2([1,1], 0.01,lam_formula_ejer2, formula_ejer2, u, v, 50, np.float64(1e-14))
    print("\nEjercicio 2.a")
    print("Encuentra un valor menor a 10^-14 en: ", int(datos_ejer2.size/3), "iteraciones contando la de (1,1)" )
    print("Se encuentran en el punto (",datos_ejer2[-1][0],",",datos_ejer2[-1][1],")")
    
    dibujarGrafica(datos_ejer2, [0.6,1.1], [0.9,1.1], 100, lam_formula_ejer2,1,"u","v","Ejercicio 2")
    
    
    f3 = implemented_function('f', lambda k,n: np.float64((k**2 + 2*n**2 + 2*sym.sin(2*np.pi*k) * sym.sin(2*np.pi*n))))
    lam_formula_ejer3 = lambdify((k,n), f3(k,n))

    x = sym.Symbol('x')
    y = sym.Symbol('y')
    formula_ejer3 = x**2 + 2*y**2 + 2*sym.sin(2*np.pi*x) * sym.sin(2*np.pi*y)
    
    datos_ejer3_a_1 = ejercicio3([0.1,0.1], 0.01,lam_formula_ejer3, formula_ejer3, x, y, 50)
    dibujarGrafica(datos_ejer3_a_1, [-0.6,1.01], [-0.5,0.4], 100, lam_formula_ejer3,2,"x","y","Ejercicio 3.a 0.01 tasa de aprendizaje")
    #dibujarGrafica(datos_ejer3_a_1, [-2,2.5], [-2,2], 100, lam_formula_ejer3,3,"x","y","Ejercicio 3.a 0.01 tasa de aprendizaje")
    
    datos_ejer3_a_2 = ejercicio3([0.1,0.1], 0.1,lam_formula_ejer3, formula_ejer3, x, y, 50)
    dibujarGrafica(datos_ejer3_a_2, [-2,2.5], [-2,2], 100, lam_formula_ejer3,4,"x","y","Ejercicio 3.a 0.1 tasa de aprendizaje")
    inicio1 = ejercicio3([1,1], 0.01,lam_formula_ejer3, formula_ejer3, x, y, 50)
    inicio2 = ejercicio3([-0.5,-0.5], 0.01,lam_formula_ejer3, formula_ejer3, x, y, 50)
    inicio3 = ejercicio3([-1,-1], 0.01,lam_formula_ejer3, formula_ejer3, x, y, 50)
    
    print("\nEjercicio 3 b")
    print("Empieza en (0.1,0.1)")
    obtenerMenor(datos_ejer3_a_1)
    print("Empieza en (1,1)")
    obtenerMenor(inicio1)
    print("Empieza en (-0.5,-0.5)" )
    obtenerMenor(inicio2)
    print("Empieza en (-1,-1)" )
    obtenerMenor(inicio3)

    
    print()
    
    
if __name__== "__main__":
  main()