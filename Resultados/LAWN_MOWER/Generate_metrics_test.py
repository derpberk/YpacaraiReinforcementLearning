#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:20:44 2020

@author: samuel
"""

# Importamos las librerias necesarias #
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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



Num_muestras = 1
v_metricas = np.asarray([]) # Vector para guardar las metricas #
v_coverage = np.asarray([])
v_mean = np.asarray([])
v_std = np.asarray([])

position_lm = genfromtxt('lawnmower_positions.csv', delimiter = ',')

Rmax = np.sum(np.sum(env.R_abs))
S = np.asarray([])
SS = np.asarray([])
SS_max = np.sum(np.sum(env.R_abs))

for k in range(0,Num_muestras):

    obs = env.reset()

    state = env.render()

    N = 300
    grid_map = env.map
    position = np.asarray([obs['position'][0],obs['position'][1]])
    position_index = int(np.where((position_lm == position).all(axis=1))[0])

    reward = 0
    num = 0
    dir = 0 # Hacia abajo

    for steps in range(N):

        if position_index == len(position_lm)-1:
            dir = 1
        elif position_index == 0:
            dir = 0
        else:
            pass

        if dir == 0:
            next_position = position_lm[position_index+1]
            position_index = position_index+1
        else:
            next_position = position_lm[position_index-1]
            position_index = position_index-1

        
        if position[0] == next_position[0] and position[1] < next_position[1]:
            action = 2
        elif position[0] == next_position[0] and position[1] > next_position[1]:
            action = 3 
        elif position[0] < next_position[0] and position[1] == next_position[1]:
            action = 1
        elif position[0] > next_position[0] and position[1] == next_position[1]:
            action = 0
        elif position[0] > next_position[0] and position[1] < next_position[1]:
            action = 4
        elif position[0] > next_position[0] and position[1] > next_position[1]:
            action = 5
        elif position[0] < next_position[0] and position[1] < next_position[1]:
            action = 6
        elif position[0] < next_position[0] and position[1] > next_position[1]:
            action = 7

        obs,rew,done,info = env.step(action)       
        position = np.asarray(obs['position'])
        
        reward += rew

    
        #state = np.dstack((obs['visited_map'],obs['importance_map']))
        state = env.render()
        S = np.append(S,reward)
        SS = np.append(SS,np.sum(np.sum(obs['importance_map'])))
            

    metrics = env.metrics()

    print("Reward: {}".format(reward))
    print("Coverage: {}".format(metrics['coverage']))
    print("Media de tiempo: {}".format(metrics['mean']))
    print("Dev. tipica: {}".format(metrics['std']))

    v_coverage = np.append(v_coverage, metrics['coverage'])
    v_mean = np.append(v_mean, metrics['mean'])
    v_std = np.append(v_std, metrics['std'])

    


v_metricas = np.column_stack((v_coverage,v_mean,v_std))


np.savetxt('metricas.csv', v_metricas , delimiter=',')

VM = obs['visited_map']
IM = np.clip(obs['importance_map'],0,None)
img = env.render()

np.savetxt('S_metricas_LW.csv', S , delimiter=',')

SS = SS/SS_max

np.savetxt('SS_metricas_LW.csv', SS, delimiter=',')

ax1.imshow(img)
ax2.imshow(VM, cmap = 'gray')
im = ax3.imshow(IM,interpolation='bicubic', cmap = 'jet_r')
divider = make_axes_locatable(ax3)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')

plt.show()

plt.imshow(np.clip(env.visited_M,None,10),interpolation='nearest')
plt.colorbar()

plt.show()


    