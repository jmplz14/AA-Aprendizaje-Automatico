# -*- coding: utf-8 -*-
"""
TRABAJO 2. 
Nombre Estudiante: 
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Fijamos la semilla
np.random.seed(1)


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna se usará una N(0,sqrt(5)) y para la segunda N(0,sqrt(7))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
        
    return out


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b

def h(x, pesos):
    solucion = np.zeros(len(x),np.float64)
    i = 0
    for i in range(len(solucion)):
        solucion[i] = np.dot(np.transpose(pesos),x[i])
        
    return solucion

def sigma(y,pesos,x):	
    valor_x=np.dot(np.transpose(pesos),x)
    return 1/(1+np.exp(-np.dot(-y,valor_x)))

def gradienteDescendenteEstocastico(x, y, num_iteraciones,tam_minibatch,tasa_aprendizaje):
    #inicio el vector de pesos
    
    pesos = np.zeros(x[0].size)
    pesos_ant = pesos
    
    
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
    paramos = False
    #reocrro hasata llegar al maximo de iteraciones
    while iteraciones < num_iteraciones and paramos == False:
        
        """
        si llevamos la posicion de minibacht igual al tamaño d eestos ya podemos
        calcular los pesos para esta pasada
        """
        if pos_minibatch == tam_minibatch:
            #Calculo los pesos
            valor = np.zeros(x[0].size)
            """for i in range(tam_minibatch):
                valor += -y_minibatch[i]*x_minibatch[i]*(1/(1+np.exp(-(-y_minibatch[i]*pesos.T*x_minibatch[i])))) 
            """
            for j in range(tam_minibatch):
                valor += np.dot((-y_minibatch[j]*x_minibatch[j]), sigma(y_minibatch[j],pesos,x_minibatch[j]))
		
            pesos = pesos_ant-tasa_aprendizaje* valor
            #print((1/tam_minibatch)/valor)
            
            """print(parentesis)
            logistico = 1/(1-np.exp(-x_minibatch)) * parentesis
            suma += np.dot(-y_minibatch,np.dot(x_minibatch,logistico))
            pesos=pesos-tasa_aprendizaje*(1/tam_minibatch)*suma
            print(pesos)"""
            
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
            """print(pesos)
            print(pesos_ant)
            print(np.linalg.norm(pesos_ant-pesos))"""
            if(np.linalg.norm(pesos_ant-pesos) < 0.01):
                paramos = True
                
            
            pos_permutacion = 0
            pos_minibatch = 0
            pesos_ant = pesos
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

#funcion de error de la regresion lineal
def error(x,y,pesos):
    total_errores = 0
    tamano = y.size
    for i in range(tamano):
        valor = 1/(1+np.exp(-np.dot(np.transpose(pesos),x[i])))
        #print(valor);

        if (valor < 0.5 and y[i] == 1):
            total_errores += 1
        if(valor > 0.5 and y[i] == 0):
            total_errores += 1
    return total_errores/tamano
    

#cargo la muestra de entrenamiento
a,b = simula_recta((0,2))
X=simula_unif(100,2,(0,2))
Y = np.sign(X.T[1] - a * X.T[0] -b)

#inicio los valores con los que quiero que ejecute
pesos = np.zeros(3)
num_iteraciones = 5000
tam_minibatch = 16
tasa_aprendizaje = 0.01

#le meto la columno de uno a los datos
X_conuno = np.ones((np.size(X,0),np.size(X,1) + 1))
X_conuno[:,1:] = X

#llamo a la funcion del gradiente con regresion lineal
pesos = gradienteDescendenteEstocastico(X_conuno, Y, num_iteraciones,tam_minibatch,tasa_aprendizaje)

#obtengo los coeficientes
a_g = (-(pesos[0]/pesos[2])/(pesos[0]/pesos[1]))
b_g = (-pesos[0]/pesos[2])

x_recta = np.linspace(0,2,100)
y_recta = a*x_recta + b
y_gradiente = a_g*x_recta + b_g


#dibujo la grafica del entrenamiento
plt.figure(figsize=(10,7))
plt.plot(x_recta,y_recta,color='blue')
plt.plot(x_recta,y_gradiente,color='red')
plt.scatter(X.T[0],X.T[1], c=Y)
plt.title('Regresión logística con los datos de entrenamiento')
plt.axis([0, 2, 0, 2])
cinco_patch = mpatches.Patch(color='yellow', label='1')
uno_patch = mpatches.Patch(color='purple', label='-1')
dos_patch = mpatches.Patch(color='blue', label='funcion')
tres_patch = mpatches.Patch(color='red', label='regresion logistica')
plt.legend(handles=[cinco_patch,uno_patch,dos_patch,tres_patch])
plt.show()

#cargo el conjunto de prueba.
x_test=simula_unif(3000,2,(0,2))
y_test = np.sign(x_test.T[1] - a * x_test.T[0] -b)

#dibujo la grafica
plt.figure(figsize=(10,7))
plt.plot(x_recta,y_recta,color='blue')
plt.plot(x_recta,y_gradiente,color='red')
plt.scatter(x_test.T[0],x_test.T[1], c=y_test)
plt.title('Regresión logística con los datos de test')
plt.axis([0, 2, 0, 2])
cinco_patch = mpatches.Patch(color='yellow', label='1')
uno_patch = mpatches.Patch(color='purple', label='-1')
dos_patch = mpatches.Patch(color='blue', label='funcion')
tres_patch = mpatches.Patch(color='red', label='regresion logistica')
plt.legend(handles=[cinco_patch,uno_patch,dos_patch,tres_patch])

plt.show()


x_test_conuno = np.ones((np.size(x_test,0),np.size(x_test,1) + 1))
x_test_conuno[:,1:] = x_test

#cambio los -1 por 0
y_test_copy = np.copy(y_test)
for i in range(y_test_copy.size):
    if(y_test_copy[i] == -1):
        y_test_copy[i] = 0

#calculo y muestro el error
print("Ejercicio 2b\n")     
print("El porcentaje de error para el test es de: ", error(x_test_conuno,y_test_copy,pesos))

###############################################################################
###############################################################################
###############################################################################
###############################################################################
print('EJERCICIO BONUS\n')

label4 = 1
label8 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 4 o la 8
	for i in range(0,datay.size):
		if datay[i] == 4 or datay[i] == 8:
			if datay[i] == 4:
				y.append(label4)
			else:
				y.append(label8)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y