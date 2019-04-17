# -*- coding: utf-8 -*-
"""
TRABAJO 2. 
Nombre Estudiante: 
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



# Fijamos la semilla
np.random.seed(16)


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

"""
datros: matriz con los datos 
label: vector de etiquetas
max_iter: numero maximo de iteraciones
vini: valor inicial del vector de pesos
devuelve los coeficientes del hiperplano
"""
def ajusta_PLA(datos, label, max_iter, vini):
    datos_perceptron = np.ones((np.size(datos,0),np.size(datos,1) + 1))
    datos_perceptron[:,1:] = datos
    pesos = vini
    pesos_ant = vini
    iteracion = 0
    num_filas = np.size(datos_perceptron,0)
    while iteracion < max_iter:
        for i in range(num_filas):
            respuesta = np.sign(np.dot(pesos.T,datos_perceptron[i]))
            if label[i] != respuesta:
                pesos = pesos + np.dot(label[i],datos_perceptron[i])
        if np.array_equal(pesos,pesos_ant):
            iteracion -= 1
            break
        pesos_ant = pesos
        
        iteracion += 1
    a = (-(pesos[0]/pesos[2])/(pesos[0]/pesos[1]))
    b = (-pesos[0]/pesos[2])
    return a,b,iteracion
                
           
             
        

        
    print(datos_perceptron)
    print(pesos)

    

 

print('Ejercicio 1-a')
a1 = simula_unif(50, 2, (-50,50))
a2 = simula_gaus(50, 2, (5,7))
plt.figure(figsize=(10,7))
plt.plot(a1,'o',markersize=4,color='green')
plt.title('Ejercicio 1a')
plt.show()

plt.figure(figsize=(10,7))
plt.plot(a2,'o',markersize=4,color='blue')
plt.title('Ejercicio 1b')
plt.show()




print('Ejercicio 2')
datos_ejer2 = simula_unif(50, 2, (-50,50))
clases = np.empty(50)
a,b = simula_recta((-50,50)) 
i = 0 
datos_transpuestos = np.transpose(datos_ejer2)
coordenada_x = datos_transpuestos[0]
coordenada_y = datos_transpuestos[1]

clases = np.sign(coordenada_y - a * coordenada_x -b)
#for x in datos_ejer2:
#    clases[i] = np.sign(x[0] - a*x[1] - b)
#    i += 1

  
x = np.linspace(-50,50,100)
y = a*x + b

plt.figure(figsize=(10,7))
plt.plot(x,y,color='blue')
plt.scatter(datos_transpuestos[0],datos_transpuestos[1], c=clases)
plt.title('Ejercicio 2a')
cinco_patch = mpatches.Patch(color='yellow', label='1')
uno_patch = mpatches.Patch(color='purple', label='-1')

plt.legend(handles=[cinco_patch,uno_patch])
plt.show()

positivos = np.zeros((clases[clases == 1].size,2))
negativos = np.zeros((clases[clases == -1].size,2))
clases_positivos =np.empty(np.size(positivos,0))
clases_negativos =np.empty(np.size(negativos,0))
i_pos = 0
i_neg = 0


for i in range(0,clases.size):
    if clases[i] == 1:
        positivos[i_pos][0] = datos_ejer2[i][0]
        positivos[i_pos][1] = datos_ejer2[i][1]
        clases_positivos[i_pos] = 1
        i_pos += 1
    else:
        negativos[i_neg][0] = datos_ejer2[i][0]
        negativos[i_neg][1] = datos_ejer2[i][1]
        clases_negativos[i_pos] = -1
        i_neg += 1
        
porcentaje_pos = int(clases_positivos.size * 0.1)
porcentaje_neg = int(clases_negativos.size * 0.1)
#posiciones_pos = random.sample(range(clases_positivos.size), porcentaje_pos)
#posiciones_neg = random.sample(range(clases_negativos.size), porcentaje_neg)

posiciones_pos = np.arange(clases_positivos.size)
np.random.shuffle(posiciones_pos)
posiciones_pos = posiciones_pos[:porcentaje_pos]

posiciones_neg = np.arange(clases_negativos.size)
np.random.shuffle(posiciones_neg)
posiciones_neg = posiciones_neg[:porcentaje_neg]

for i in posiciones_pos:
    clases_positivos[i] = -1

for i in posiciones_neg:
    clases_negativos[i] = 1

clases_2b = np.concatenate([clases_positivos,clases_negativos])
datos_2b = np.concatenate([positivos,negativos])

positivos = np.transpose(positivos)
negativos = np.transpose(negativos)



plt.figure(figsize=(10,7))
plt.plot(x,y,color='blue')
plt.scatter(positivos[0],positivos[1], c=clases_positivos)
plt.scatter(negativos[0],negativos[1], c=clases_negativos)
plt.title('Ejercicio 2B')
cinco_patch = mpatches.Patch(color='yellow', label='1')
uno_patch = mpatches.Patch(color='purple', label='-1')
plt.legend(handles=[cinco_patch,uno_patch])
plt.show()

#ejercicio 3
#primera formula

x = np.linspace(-50,50,100)
y = np.linspace(-50,50,100)
x, y = np.meshgrid(x, y)
z = (x-10)**2 + (y-20)**2 - 400
#z = (x-10)**2 + (y-20)**2 - 400

plt.figure(figsize=(10,7))
plt.contour(x,y,z,[0])
plt.scatter(positivos[0],positivos[1], c=clases_positivos)
plt.scatter(negativos[0],negativos[1], c=clases_negativos)
plt.title('Ejercicio 3 formula (x-10)**2 + (y-20)**2 - 400 ')
cinco_patch = mpatches.Patch(color='yellow', label='1')
uno_patch = mpatches.Patch(color='purple', label='-1')
plt.legend(handles=[cinco_patch,uno_patch])
plt.show()

z = 0.5*(x+10)**2 + (y-20)**2 - 400
plt.figure(figsize=(10,7))
plt.contour(x,y,z,[0])
plt.scatter(positivos[0],positivos[1], c=clases_positivos)
plt.scatter(negativos[0],negativos[1], c=clases_negativos)
plt.title('Ejercicio 3 formula 0.5*(x+10)**2 + (y-20)**2 - 400 ')
cinco_patch = mpatches.Patch(color='yellow', label='1')
uno_patch = mpatches.Patch(color='purple', label='-1')
plt.legend(handles=[cinco_patch,uno_patch])
plt.show()

z = 0.5*(x-10)**2 - (y+20)**2 - 400
plt.figure(figsize=(10,7))
plt.contour(x,y,z,[0])
plt.scatter(positivos[0],positivos[1], c=clases_positivos)
plt.scatter(negativos[0],negativos[1], c=clases_negativos)
plt.title('Ejercicio 3 formula 0.5*(x+10)**2 + (y-20)**2 - 400 ')
cinco_patch = mpatches.Patch(color='yellow', label='1')
uno_patch = mpatches.Patch(color='purple', label='-1')
plt.legend(handles=[cinco_patch,uno_patch])
plt.show()

z = y - 20*x**2 - 5*x + 3
plt.figure(figsize=(10,7))
plt.contour(x,y,z,[0])
plt.scatter(positivos[0],positivos[1], c=clases_positivos)
plt.scatter(negativos[0],negativos[1], c=clases_negativos)
plt.title('Ejercicio 3 formula 0.5*(x+10)**2 + (y-20)**2 - 400 ')
cinco_patch = mpatches.Patch(color='yellow', label='1')
uno_patch = mpatches.Patch(color='purple', label='-1')
plt.legend(handles=[cinco_patch,uno_patch])
plt.show()

#2 modelos lineales.
print("\nEjercicios Regresion Lineal\n")

pesos = np.zeros(np.size(datos_ejer2,1)+1)
a_1a,b_1a,num_iter = ajusta_PLA(datos_ejer2,clases,100,pesos)
print("Ejercicio 1 apartado a\n")
print("Con el vector iniciado a 0 se necesitan ", num_iter , " iteraciones\n" )
x = np.linspace(-50,50,100)
y = a_1a*x + b_1a

plt.figure(figsize=(10,7))
plt.plot(x,y,color='blue')
plt.scatter(datos_ejer2.T[0],datos_ejer2.T[1], c=clases)
plt.title('Ejercicio 1a Regresion lineal')
cinco_patch = mpatches.Patch(color='yellow', label='1')
uno_patch = mpatches.Patch(color='purple', label='-1')

plt.legend(handles=[cinco_patch,uno_patch])
plt.show()

a_1b = 0
b_1b = 0
num_iter_total = 0
for i in range(10):
    pesos = np.random.rand(np.size(datos_ejer2,1)+1)
    a_1b,b_1b,num_iter = ajusta_PLA(datos_ejer2,clases,150,pesos)
    num_iter_total += num_iter

print("Con el vector iniciado aleatoriamente se necesitan ", num_iter_total/10 , " iteraciones\n" ) 

pesos = np.zeros(np.size(datos_ejer2,1)+1)
a_2b,b_2b,num_iter = ajusta_PLA(datos_2b,clases_2b,250,pesos)

x = np.linspace(-50,50,100)
y = a_2b*x + b_2b
plt.figure(figsize=(10,7))
plt.plot(x,y,color='blue')
plt.scatter(datos_2b.T[0],datos_2b.T[1], c=clases_2b)
plt.title('Ejercicio 1b Regresion lineal')
cinco_patch = mpatches.Patch(color='yellow', label='1')
uno_patch = mpatches.Patch(color='purple', label='-1')

plt.legend(handles=[cinco_patch,uno_patch])
plt.show()

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