#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:47:24 2020

Test de verificación del entorno creado. Se cargará el entorno diseñado y se
comprobará que funcionan de forma adecuada todas las funciones necesarias para 
el efectivo entrenamiento de la tarea que se quiere llevar a cabo.

@author: samuel
"""

# Importamos las librerias necesarias #
import numpy as np
import matplotlib.pyplot as plt
import YpacaraiMap
from time import sleep


# Cargamos el escenario #

env = YpacaraiMap.Environment()

# Creamos una ventana para realizar el renderizado #


# Reseteamos el entorno #
" Habría que introducir que al resetear el barco surja en un pto. aleatorio."
env.reset()

# Creamos los estimulos que le vamos a dar al robot (bucle abierto) 
N = 1000 # Numero de accionea aleatorias
acciones = np.random.randint(0,7,size = N)

# Bucle de las acciones #

fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(16,8))

ax1.set_xticks(np.arange(env.S.shape[1])-0.5)
ax1.set_yticks(np.arange(env.S.shape[0])-0.5)
ax1.grid(True, color = np.asarray([0,110,5])/255,linewidth = 1, alpha = 0.4)
ax1.axes.xaxis.set_ticklabels([])
ax1.axes.yaxis.set_ticklabels([])

ax2.set_xticks(np.arange(env.S.shape[1]))
ax2.set_yticks(np.arange(env.S.shape[0]))
ax2.grid(True, linewidth = 0.5, alpha = 0.1, drawstyle = 'steps-mid')
plt.setp(ax2.get_xticklabels(), rotation_mode="anchor")

ax3.set_xticks(np.arange(env.S.shape[1]))
ax3.set_yticks(np.arange(env.S.shape[0]))
ax3.grid(True, linewidth = 0.5, alpha = 0.2, drawstyle = 'steps-mid')
plt.setp(ax3.get_xticklabels(), rotation_mode="anchor")

fig.suptitle('Mapas de estado del barco')

for i in range(acciones.size):
    
    print("Se va a realizar una acción tipo {}".format(acciones[i]))
    
    obs,rew,done,info = env.step(acciones[i])
    
    print("La recompensa de esta acción ha sido: {0:.3f}".format(rew))
    print("OBS toma el valor de [X,Y] = [{},{}]".format(obs['position'][0],obs['position'][1]))
        

print("Ciclo terminado con éxito.")

VM = obs['visited_map']
IM = obs['importance_map']

ax1.imshow(env.render())
ax2.imshow(VM, cmap = 'gray')
ax3.imshow(IM,interpolation='bicubic', cmap = 'jet_r')

plt.show()







