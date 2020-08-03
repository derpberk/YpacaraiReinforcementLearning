#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:15:50 2020

@author: samuel
"""

import DDQNAgent
import numpy as np
import matplotlib.pyplot as plt
import YpacaraiMap
from torch import save

from tqdm import tqdm

# Definimos los hiperparámetros #

steps = 300
epochs = 1500
gamma = 0.95
epsilon = 0.99
lr = 1e-3
n_actions = 8
input_dims = (3,25,19)
mem_size = 10000
batch_size = 250
eps_min = 0.01
eps_dec = (epsilon-eps_min)/1400
replace = 50

# Creamos el agente #

agente = DDQNAgent.DDQNAgent(gamma, epsilon, lr, n_actions, input_dims, 
                 mem_size, batch_size, eps_min, eps_dec, replace)

# Inicializamos el escenario #

env = YpacaraiMap.Environment()

# Wrapper para seleccionar qué es el estado #
def do_step(env,action):
    
    obs, reward, done, info = env.step(action)
       
    state = env.render()
    
    return state, reward, done, info
    
def reset(env):
    
    env.reset()
       
    state = env.render()
    
    return state

# Semillas #
np.random.seed(42)

# Creamos la figura #

fig = plt.figure(figsize=(8, 4))
fig.show()
fig.canvas.draw()
plt.xlim([0,epochs])
plt.grid(True, which = 'both')

filtered_reward = 0
filtered_reward_buffer = []
reward_buffer = []

record = -100000

# Comenzamos el entrenamiento #

for epoch in tqdm(range(0,epochs)):
    
    state = reset(env)
    rew_episode = 0

    # Mermamos epsilon#
    agente.decrement_epsilon()
    
    for step in range(steps):
        
        # Llamamos a la política de comportamiento #
        action = agente.choose_action_epsilon_greedy(state)
        
        # Aplicamos la acción escogida #
        next_state, reward, done, info = do_step(env,action)
        
        # Guardamos la experiencia #
        agente.store_transition(state,action,reward,next_state,done)
        
        # El estado anterior pasa a ser el actual #
        state = next_state
        
        # Acumulamos la recompensa total #
        rew_episode += reward
        
        # Entrenamos. Si no hay suficientes experiencias, learn() retorna #
        agente.learn()
        
    # Actualizamos la red del target (si es que toca) #
    agente.replace_target_network(epoch)

    if epoch == 0:
        filtered_reward = rew_episode
    else:
        filtered_reward = rew_episode*0.05 + filtered_reward*0.95
    
    reward_buffer.append(rew_episode)
    filtered_reward_buffer.append(filtered_reward)

    if(record < rew_episode):
        print('Nuevo record de {:06.2f} en el episodio {:d}\n'.format(rew_episode,epoch))
        record = rew_episode
        save(agente.q_eval, "DDQN_BEST.pt")

    # Dibujamos la recompensa #
    plt.plot(reward_buffer,'b',alpha=0.2)
    plt.plot(filtered_reward_buffer,'r')
    plt.pause(0.001)
    fig.canvas.draw()
    

print('Entrenamiento terminado!')

save(agente.q_eval, "DDQN_LAST.pt")
        
        

