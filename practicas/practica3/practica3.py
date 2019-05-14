#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:23:46 2019

@author: jose
"""
from numpy import genfromtxt
import numpy as np

def normalizarDatos(datos):
    

def loadCSV(fichero):
    
    my_data = genfromtxt(fichero, delimiter=',')
    clases = my_data[:, -1] 
    datos = my_data[:, :-1]
    
    return datos,clases




def main():
    train_X, train_y = loadCSV("datos/optdigits.tra")
    test_X, text_y = loadCSV("datos/optdigits.tes")
 
    
if __name__== "__main__":
  main()
  
  
