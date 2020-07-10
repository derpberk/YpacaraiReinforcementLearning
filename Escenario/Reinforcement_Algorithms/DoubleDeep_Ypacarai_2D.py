#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25

@author: samuel
"""


# Importamos las bibliotecas de propósito general #
import numpy as np
from collections import deque
from tqdm import tqdm

# Bibliotecas de NNs (Keras y Tensorflow)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten, Conv2D, LSTM, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# Importamos nuestro escenario custom #
import YpacaraiMap

# Eliminamos la sesión anterior de Keras por si acaso #
clear_session()

# Cargamos el escenario 
env = YpacaraiMap.Environment()

# Lo reseteamos #
env.reset()

# Wrapper para seleccionar qué es el estado #
def do_step(env,action):
    
    obs, reward, done, info = env.step(action)
       
    state = env.render()
    
    return state, reward, done, info
    
def reset(env):
    
    env.reset()
       
    state = env.render()
    
    return state    

class Agente:

    """ 
    Clase agente en la que se engloban las funciones de entrenamiento,
    la red neuronal y demás.
    """
    
    def __init__(self,env,optimizers): # Constructor de la clase #

        # Inicializamos los los atributos del la estructura del problema #
        # Tamaño del estado #
        self._state_size = (25,19,3)
        # Tamaño de la entrada #
        self._action_size = 8
        # Optimizador para la DNN #
        self._optimizer = optimizer 
        # Función de coste utilizada #
        self.loss_fn = keras.losses.Huber()
        # Inicialización del replay memory #
        self.replay_memory = deque(maxlen = 10000)
        # Parámetros del soft-updating
        self.tau = 0.125
        self.soft_update = 1 # ACTIVADO. Si está desactivado tenemos un DDQN normal. #
        # Cada cuantos episodios se actualiza la red del target #
        self.target_episode_update_freq = 100

        # Hiperparametros del Reinforcement Learning #
        # Con discount rate = 0 -> miope. Con discount rate -> 1 -> largoplacista #
        self.discount_rate = 0.95 # Ponderación de la recompensa futura frente a la inmediata #
        self.epsilon = 0.995

        # Creacion de la red #
        self.model, self.target = self._build_compile_model() # Creamos las DNN (las 2) #
        
    # Ahora pasamos a definir los métodos del agente. Primero el método 
    # para acumular una experiencia en el memory replay: 

    # Funcion de guardado en memoria de un step #
    def append_memory(self, state, action, reward, next_state, done):
        self.replay_memory.append((state,action,reward,next_state,done))

    # Método donde creamos la DNN para estimar tanto función de target y la funcion Q: 
    # Función de creación de la DNN #      
    def _build_compile_model(self):
        
        # Modelo 1 - Only Conv2D #

        model = Sequential()
        model.add(Conv2D(8, kernel_size=(5, 5), strides=(2, 2),
                          activation='elu',
                          input_shape=(25,19,3)))
        model.add(Flatten())
        model.add(Dense(1024, activation='elu'))
        model.add(Dense(1024, activation='elu'))
        model.add(Dense(8, activation='linear'))
    
        
        model.compile(loss = 'Huber', optimizer = self._optimizer)
        
        target = keras.models.clone_model(model)
        target.set_weights(model.get_weights())
        
        return model, target
        
    # Toma una decisión de qué acción tomar siguiendo la policy e-greedy #
    def take_action(self,state):

        # Estrategia e-greedy #
        if np.random.rand() <= self.epsilon:
            return env.action_space_sample()
        else:
            q_values = self.model.predict(state[np.newaxis])
            return np.argmax(q_values[0])

    # Para muestrear una experiencia. Sacado del script de dguti #
    def sample_experiences(self,batch_size):
        indices = np.random.randint(len(self.replay_memory), size=batch_size)
        batch = [self.replay_memory[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)]
        return states, actions, rewards, next_states, dones

    def make_a_training_step(self,batch_size):
          
        # Muestreamos un batch de experiencias #
        experiences = self.sample_experiences(batch_size)

        # Volcamos los valores #
        states, actions, rewards, next_states, dones = experiences
            

        #next_states = next_states.reshape(-1, 1)
        
        # Predecimos los valores de Q(s', cada acción) con la FUNCION TARGET
        next_Q_values = self.model.predict(next_states)
        
        best_next_actions = np.argmax(next_Q_values, axis = 1)
        
        next_mask = tf.one_hot(best_next_actions, self._action_size).numpy()
    
        # Calculamos el target #
        max_next_Q_values = (self.target.predict(next_states) * next_mask).sum(axis=1)


        # Evaluamos la función de target - si está done no aplicamos la tasa de
        # descuento, puesto que la experiencia está terminada en el tiempo. #
        target_Q_values = (rewards +  self.discount_rate * max_next_Q_values) 

        # Reshape sobre los 
        target_Q_values = target_Q_values.reshape(-1, 1)

        # Máscara 
        # One Hot encoding es esto:
        # [0] -> [1,0,0,0,0,0]
        # [1] -> [0,1,0,0,0,0]
        # [2] -> [0,0,1,0,0,0]
        # [3] -> [0,0,0,1,0,0]
        # ... etc
        mask = tf.one_hot(actions, self._action_size)

        # Método tape para grabar el forwarding de la red y calcular el gradient
        # de esa operación para aplicar el gradiente descendiente.

        with tf.GradientTape() as tape: # Calculamos el gradiente
            all_Q_values = self.model(states) # Se llama al modelo actual
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            # Calculamos la funcióón de loss
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        # Computamos el gradiente de la operación sobre el coste #
        grads = tape.gradient(loss, self.model.trainable_variables) # paso del gradiente
        # Aplicamos el gradiente ajustando los parámetros #
        self._optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        # ¿No se podría utilizar la función model.fit() ... ?
        return loss
    
    def update_target(self,episode):
        
        # Si no tenemos soft update...
        if self.soft_update == 0:
            # Volvamos la red directamente cada target_episode_update_freq episodios #
            if(episode%self.target_episode_update_freq == 0):
                Barco.target.set_weights(Barco.model.get_weights())
                print("Se ha volcado la red del modelo en la red del target.")
            else:
                pass
        # Si tenemos soft update, pasamos los parámetros filtrados #
        else:
            weights = self.model.get_weights()
            target_weights = self.target.get_weights()
            for i in range(len(target_weights)):
                target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
                self.target.set_weights(target_weights)
        


# Entrenamiento #

optimizer = Adam(learning_rate = 0.001)
Barco = Agente(env,optimizer)

np.random.seed(42)
tf.random.set_seed(42)

"""Establecemos los parámetros de entrenamiento que seráán 3:


1.   El tamaño del batch de entrenamiento.
2.   Cuántos episodios (partidas) jugará.
3.   Cuántos movimientos dura como máximo la partida.
"""

# Parámetros de entrenamiento #

batch_size = 250
num_of_episodes = 1500
steps_per_episode = 200
best_score = -100000 # Cota máxima - infinito #
r_buffer = []
fr_buffer = []
epi_buffer = []
filtered_reward = 0
first = 1
loss = -1

"""Mostramos un resumen de la red"""

Barco.model.summary()
Barco.target.summary()

"""**Ejecutamos el entrenamiento por fin** (OJO que esto es lo que tarda)"""

fig = plt.figure(figsize=(16, 8))
fig.show()
fig.canvas.draw()
plt.xlim([0,num_of_episodes])

for episode in tqdm(range(0,num_of_episodes),desc='Rec: {:.2f} '.format(filtered_reward)):

    # Resetamos el entorno #

    state = reset(env)
    
    rew_episode = 0
    
    epi_buffer.append(episode+1)

    # Comenzamos la ejecución de un episodio #
    for step in range(steps_per_episode):
        
        
        # Mermamos epsilon #
        Barco.epsilon = max(1 - episode / num_of_episodes, 0.1)
        # Computamos qué acción tomamos #
        action = Barco.take_action(state)

        # La aplicamos y observamos qué repercusión tiene #
        next_state, reward, done, info = do_step(env,action)
        
        # Guardamos el resultado de esa acción #
        Barco.append_memory(state,action,reward,next_state,done)

        # El estado anterior pasa a ser el actual #
        state = next_state

        # Acumulamos la recompensa total #
        rew_episode += reward

        # Si terminamos #
        if done:
            break

        # Cuando tengamos suficientes datos, esto es -> replay_mem > batch_size ...
        # ya podemos entrenar la red #
        if len(Barco.replay_memory) > batch_size:
            loss = Barco.make_a_training_step(batch_size)
        # if step % 10 == 0:    
        #     print("Episode: {}, Steps: {}, Rew: {}\n".format(episode, step,rew_episode),end="") # Not shown
    
    # Cargamos en el buffer la recompensa del episodio que se acaba de ejecutar #
    r_buffer.append(rew_episode) 
    
    if first == 1:
        filtered_reward = rew_episode
        first = 0
    else:
        filtered_reward = 0.95*filtered_reward + rew_episode*0.05
        
    fr_buffer.append(filtered_reward)

    if rew_episode > best_score: # Si batimos el record
        best_weights = Barco.model.get_weights() # Guardamos la DNN que bate el record
        best_score = rew_episode # Actualizamos el record
        print("\nEP: {} --- Nuevo record con Reward de: {} y valor de loss {:06.2f}".format(episode,best_score,loss))
    
    # Soft Update del la red del target. Se le pasa el episodio para comprobar si se vuelva o no en el caso de que no haya soft-update #
    Barco.update_target(episode)
        
    # Dibujamos #  
    plt.plot(epi_buffer,r_buffer,'b',alpha=0.2)
    plt.plot(epi_buffer,fr_buffer,'r')
    plt.grid(True, which = 'both')
    plt.pause(0.001)
    fig.canvas.draw()
    
        
        
"""Cuando termine de entrenar, tomamos los pesos de la red neuronal entrenada que ha dado el mejor resultado y lo guardamos."""

# Cargamos en la red el mejor modelo entrenado #
# Guardamos la ultima #
Barco.model.save("DoubleDeep2D_Ypacarai_Model_LAST.h5")
Barco.model.set_weights(best_weights)
# Guardamos la red con los pesos #
Barco.model.save("DoubleDeep2D_Ypacarai_Model_BEST.h5")
    
print("\n Terminado! \n")

np.savetxt("reward_DoubleDQN_2D.csv", r_buffer, delimiter=",")
plt.figure(figsize=(16, 8))

plt.plot(epi_buffer,fr_buffer,'r')
error = np.abs(np.asarray(fr_buffer) - np.asarray(r_buffer))
plt.fill_between(epi_buffer,fr_buffer+error,fr_buffer-error,color = 'blue', alpha=0.3)
plt.grid(True, which = 'both')

plt.xlabel("Episode", fontsize=14)
plt.ylabel("Sum of rewards", fontsize=14)
plt.show()
