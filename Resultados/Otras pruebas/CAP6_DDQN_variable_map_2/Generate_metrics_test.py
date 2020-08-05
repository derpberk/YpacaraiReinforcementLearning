#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:20:44 2020

@author: samuel
"""

# Importamos las librerias necesarias #
import numpy as np
from time import sleep
import random
import os
from IPython.display import clear_output
from collections import deque

# Bibliotecas de NNs (Keras y Tensorflow)
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape,Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Importamos nuestro escenario custom #
import YpacaraiMap

# Cargamos el escenario 
env = YpacaraiMap.Environment()
env.set_test_mode(True)

fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(16,8))

ax2.set_xticks(np.arange(env.S.shape[1]))
ax2.set_yticks(np.arange(env.S.shape[0]))
ax2.grid(True, linewidth = 0.5, alpha = 0.1, drawstyle = 'steps-mid')
plt.setp(ax2.get_xticklabels(), rotation_mode="anchor")


ax3.set_xticks(np.arange(env.S.shape[1]))
ax3.set_yticks(np.arange(env.S.shape[0]))
ax3.grid(True, linewidth = 0.5, alpha = 0.2, drawstyle = 'steps-mid')

plt.setp(ax3.get_xticklabels(), rotation_mode="anchor")

fig.suptitle('Mapas de estado del barco')


# Eliminamos la sesión anterior de Keras por si acaso #
clear_session()


# Wrapper para seleccionar y componer el estado como las dos matrices
def do_step(env,action):
    
    obs, reward, done, info = env.step(action)
       
    state = np.dstack((obs['visited_map'],obs['importance_map']))
    
    return state, reward, done, info
    
def reset(env):
    
    obs = env.reset()
       
    state = np.dstack((obs['visited_map'],obs['importance_map']))
    
    return state    

Num_muestras = 10
v_metricas = np.asarray([]) # Vector para guardar las metricas #
v_coverage = np.asarray([])
v_mean = np.asarray([])
v_std = np.asarray([])

for k in range(0,Num_muestras):

    # Ejecutamos la mejor de las partidas #
    obs = env.reset()

    state = env.render()

    N = 500
    grid_map = env.map
    position = obs['position']

    model = keras.models.load_model('./DQN2D_Ypacarai_Model_BEST.h5')
    reward = 0
    num = 0

    for steps in range(N):
    
        # Predicción y accion #
        q_values = model.predict(state[np.newaxis])
        
        if np.random.rand()<0.9:
            action = np.argmax(q_values[0])
        else:
            action = np.random.randint(0,8)
            
        valid = 0
        while valid == 0:
            
            if action == 0: # NORTH
                if grid_map[position[0]-1][position[1]] == 0:
                    valid = 0
                else:
                    valid = 1
                    
            elif action == 1: # SOUTH
                if grid_map[position[0]+1][position[1]] == 0:
                    valid = 0
                else:
                    valid = 1
                    
            elif action == 2: # EAST
                if grid_map[position[0]][position[1]+1] == 0:
                    valid = 0
                else:
                    valid = 1
                    
            elif action == 3: # WEST
                if grid_map[position[0]][position[1]-1] == 0:
                    valid = 0
                else:
                    valid = 1
                    
            elif action == 4: # NE
                if grid_map[position[0]-1][position[1]+1] == 0:
                    valid = 0
                else:
                    valid = 1
                    
            elif action == 5: # NW
                if grid_map[position[0]-1][position[1]-1] == 0:
                    valid = 0
                else:
                    valid = 1
                    
            elif action == 6: # SE
                if grid_map[position[0]+1][position[1]+1] == 0:
                    valid = 0
                else:
                    valid = 1
                    
            elif action == 7: # SW
                if grid_map[position[0]+1][position[1]-1] == 0:
                    valid = 0
                else:
                    valid = 1
                    
            if valid == 0:
                action = np.random.randint(0,8)
                
                
        obs,rew,done,info = env.step(action)       
        position = obs['position']
        
        reward += rew
    
        #state = np.dstack((obs['visited_map'],obs['importance_map']))
        state = env.render()
            

    metrics = env.metrics()

    print("Coverage: {}".format(metrics['coverage']))
    print("Media de tiempo: {}".format(metrics['mean']))
    print("Dev. tipica: {}".format(metrics['std']))

    v_coverage = np.append(v_coverage, metrics['coverage'])
    v_mean = np.append(v_mean, metrics['mean'])
    v_std = np.append(v_std, metrics['std'])


v_metricas = np.column_stack((v_coverage,v_mean,v_std))


np.savetxt('metricas.csv', v_metricas , delimiter=',')
