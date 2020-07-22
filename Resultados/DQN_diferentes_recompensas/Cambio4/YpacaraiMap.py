#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 12:34:59 2020

@author: samuel
"""

import numpy as np
from numpy import genfromtxt

# Entorno custom más eficiente #

# Clase del entorno #
class Environment:
    
    def __init__(self):
        
        # Cargamos el mapa y creamos las matrices de estado#
        self.map = genfromtxt('./YpacaraiMap.csv', delimiter=',',dtype = int)
        self.S = np.ones(self.map.shape)
        self.R_abs = np.ones(self.map.shape)*255
        self.visited = np.zeros(self.map.shape)
        
        # Mapas para las metricas #
        self.visited_M = np.zeros(shape = self.map.shape)
        self.waiting_M = np.ones(shape = self.map.shape)
        self.mean_time_M = np.zeros(shape = self.map.shape)
        
        # El test mode pone el mapa en modo test y no para entrenar #
        # Se realizan métricas extras no necesarias en el entrenamiento y que podrían
        # realentizarlo #
        self.test_mode = False
        
        # Seleccionamos un punto aleatorio de comienzo #
        posible_x, posible_y = np.nonzero(self.map)
        
        init_cell_index = np.random.randint(0,posible_x.size)
        
        self.agent_start_pos = (posible_x[init_cell_index],posible_y[init_cell_index])

        self.agent_pos = self.agent_start_pos
        self.agent_pos_ant = self.agent_start_pos
        
        # Marcamos el inicio como visitado #
        
        self.visited[self.agent_pos[0]][self.agent_pos[1]] = 255
        self.S[self.agent_pos[0]][self.agent_pos[1]] = 0
        self.R_abs[self.agent_pos[0]][self.agent_pos[1]] -= 50
        
    def reset(self):
        # Reseteamos las matrices de estado #
        
        self.S = np.ones(self.map.shape)
        self.R_abs = np.ones(self.map.shape)*255
        self.visited = np.zeros(self.map.shape)
        
        # Mapas para las metricas #
        self.visited_M = np.zeros(shape = self.map.shape)
        self.waiting_M = np.ones(shape = self.map.shape)
        self.mean_time_M = np.zeros(shape = self.map.shape)
        
        # El test mode pone el mapa en modo test y no para entrenar #
        # Se realizan métricas extras no necesarias en el entrenamiento y que podrían
        # realentizarlo #
        
        # Seleccionamos un punto aleatorio de comienzo #
        posible_x, posible_y = np.nonzero(self.map)
        
        init_cell_index = np.random.randint(0,posible_x.size)
        
        self.agent_start_pos = [posible_x[init_cell_index],posible_y[init_cell_index]]

        self.agent_pos = self.agent_start_pos
        self.agent_pos_ant = self.agent_start_pos
        
        # Marcamos el inicio como visitado #
        self.visited[self.agent_pos[0]][self.agent_pos[1]] = 255
        self.S[self.agent_pos[0]][self.agent_pos[1]] = 0
        
        # Calculamos el estado #
        obs = {}
        obs['visited_map'] = self.visited
        obs['importance_map'] = self.S*self.R_abs
        obs['position'] = self.agent_pos
        
        return obs
        
    def step(self,action):

        future_pos = np.copy(self.agent_pos_ant)
        
        if action == 0: # North
             future_pos[0] -= 1
        elif action == 1: # South
             future_pos[0] += 1
        elif action == 2: # East #
             future_pos[1] += 1
        elif action == 3: # West #
             future_pos[1] -= 1
        elif action == 4: # NE #
             future_pos[0] -= 1
             future_pos[1] += 1
        elif action == 5: # NW #
             future_pos[0] -= 1
             future_pos[1] -= 1
        elif action == 6: # SE #
             future_pos[0] += 1
             future_pos[1] += 1
        elif action == 7: # SW #
             future_pos[0] += 1
             future_pos[1] -= 1
            
        # Comprobamos si es un movimiento ilegal #
        ilegal = 0
        if self.map[future_pos[0]][future_pos[1]] == 0:
            ilegal = 1
        else:
            self.agent_pos = future_pos
            
        # La posicion anterior se marca en el mapa de visitacion #
        self.visited[self.agent_pos_ant[0]][self.agent_pos_ant[1]] = 127 # Casilla anterior sombreada !
        self.visited[self.agent_pos[0]][self.agent_pos[1]] = 255 # Casilla visitada bien marcada!
        
        # Se purga el interés de la casilla de la que venimos # #
        self.R_abs[self.agent_pos_ant[0]][self.agent_pos_ant[1]] =  np.max([self.R_abs[self.agent_pos_ant[0]][self.agent_pos_ant[1]]-50,0])
        
        # Procesamos el reward #
        
        rho_next = self.R_abs[self.agent_pos[0]][self.agent_pos[1]] * self.S[self.agent_pos[0]][self.agent_pos[1]]
        rho_act = self.R_abs[self.agent_pos_ant[0]][self.agent_pos_ant[1]] * self.S[self.agent_pos_ant[0]][self.agent_pos_ant[1]]
        
        # Calculamos la recompensa como el gradiente de interés #
        
        reward = rho_next - rho_act
        # Ojito que aquí decidimos cuánto penalizamos visitar una ilegal una anterior o una nueva #
        reward = (1-ilegal)*((2.507/255)*(reward-255)+1.5) - ilegal*(5)
        
        # Actualizamos la matriz S #
        
        
        for i in range(0,self.map.shape[0]):
            for j in range(0,self.map.shape[1]):
                
                self.S[i][j] = np.min([self.S[i][j]+0.0055, 1])
                
        self.S[self.agent_pos[0]][self.agent_pos[1]] = 0
        
        # Actualizamos la posición #
        self.agent_pos_ant = self.agent_pos
        
        
        obs = {}
        obs['visited_map'] = self.visited
        obs['importance_map'] = self.S*self.R_abs
        obs['position'] = self.agent_pos
        
        # Nunca acabamos - Non episodic #
        done = 0
        
        # PARA TEST_MODE ACTIVADO #
        if(self.test_mode == True):
            # Aumentamos una visita en la celda correspondiente #
            self.visited_M[self.agent_pos[0]][self.agent_pos[1]] += 1
            # Tomamos el tiempo que se ha tardado en visitar la celda esa vez #
            T = self.waiting_M[self.agent_pos[0]][self.agent_pos[1]]
            # Aumentamos el tiempo de todas las celdas menos la que acabamos de visitar #
            self.waiting_M +=  1
            self.waiting_M[self.agent_pos[0]][self.agent_pos[1]] = 1
            # Acumulamos el tiempo de visita de la celda #
            self.mean_time_M[self.agent_pos[0]][self.agent_pos[1]] += T
            
        else:
            pass
        
        return obs, reward, done, ilegal
    
    # Metodo para renderizar con colores el mapa #
    def render(self):
        
        green_color = np.asarray([0,160,20])/255
        blue_color = np.asarray([0,0,0])/255
        agent_color = np.asarray([255,0,0])/255
        red_color = np.asarray([241,241,241])/255
        
        # Hacemos una copia del mapa #
        
        size_map = (self.map.shape[0],self.map.shape[1],3)
        base_map = np.zeros(size_map)
        
        for i in range(0,self.map.shape[0]):
            for j in range(0,self.map.shape[1]):
                
                if(self.map[i][j] == 0):
                    base_map[i][j] = green_color
                elif(self.agent_pos[0] == i and self.agent_pos[1] == j):
                    base_map[i][j] = agent_color
                elif(self.visited[i][j] != 0):
                    state_of_visited = (255-self.R_abs[i][j]*self.S[i][j])/255               
                    red_color = [state_of_visited,state_of_visited,state_of_visited]
                    base_map[i][j] = red_color
                else:
                    base_map[i][j] = blue_color
        
        return base_map
    
    def action_space_sample(self):
        
        return np.random.randint(0,8)

    def set_test_mode(self,test_mode = False):
        self.test_mode = test_mode

        if(test_mode == True):
            print("Modo test activado")
        else:
            print("Modo test desactivado")

    
    
    # Metodo que devuelve las metricas #
    def metrics(self):
        
        if self.test_mode == False:
            print("El modo test no se ha activado. Esta llamada no tendrá efecto")
            return -1
        
        # Metrica de coverage #
        
        num_celdas_cubiertas = np.count_nonzero(self.visited)
        num_celdas_visitables = np.count_nonzero(self.map)
        
        coverage = num_celdas_cubiertas/num_celdas_visitables
        
        # Metrica de la frecuencia media de visitas #
        
        V = []
        # obtenemos el vector de tiempo de espera de cada visita#
        T_mean_M = self.mean_time_M/self.visited_M
        
        for i in range(0,self.map.shape[0]):
            for j in range(0,self.map.shape[1]):
                
                if(self.map[i][j] == 1 and T_mean_M[i][j] != np.inf and T_mean_M[i][j] > 0):
                    V.append(T_mean_M[i][j])
        
        
        # Calculamos la media #                    
        mean = np.mean(V)
        
        # La desviación tipica #
        std = np.std(V)
        
        metricsD = {}
        metricsD['coverage'] = coverage
        metricsD['mean'] = mean
        metricsD['std'] = std
        metricsD['time_matrix'] = T_mean_M
        metricsD['visited_acc_matrix'] = self.visited_M
        metricsD['V'] = V
        
        return metricsD
    
        
        
        
        
        
                    
                
        
        
        
        
        
        
        
            
        