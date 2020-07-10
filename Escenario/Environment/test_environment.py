#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:14:16 2020

@author: samuel
"""


# Importamos las librerias necesarias #
import numpy as np
import YpacaraiMap
import curses
import matplotlib.pyplot as plt

# Cargamos el escenario #

env = YpacaraiMap.Environment()


# Reseteamos el entorno #
" Habría que introducir que al resetear el barco surja en un pto. aleatorio."
env.reset()


# get the curses screen window
screen = curses.initscr()
 
# turn off input echoing
curses.noecho()
 
# respond to keys immediately (don't wait for enter)
curses.cbreak()
 
# map arrow keys to special values
screen.keypad(True)

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

#ax2.set_xticklabels(np.arange(0,len(env.S.shape[0])))
#ax2.set_yticklabels(np.arange(0,len(env.S.shape[1])))

ax3.set_xticks(np.arange(env.S.shape[1]))
ax3.set_yticks(np.arange(env.S.shape[0]))
ax3.grid(True, linewidth = 0.5, alpha = 0.2, drawstyle = 'steps-mid')
plt.setp(ax3.get_xticklabels(), rotation_mode="anchor")

fig.suptitle('Mapas de estado del barco')

try:
    
    while True:

        # Actualizamos la posición actual #
        
        char = screen.getch()
        if char == ord('q'):
            break
        elif char == ord('r'):
            print("Reseteando")
            obs = env.reset()
            VM = obs['visited_map']
            IM = obs['importance_map']
            
            ax2.imshow(VM.T, cmap = 'gray')
            ax3.imshow(IM.T,interpolation='bicubic', cmap = 'jet')
            plt.pause(0.002)
            fig.show()
            continue
        elif char == ord('6'):
            # print doesn't work with curses, use addstr instead
            print('DERECHA\n\r')
            accion = 2
        elif char == ord('4'):
            print('IZQUIERDA\n\r')     
            accion = 3
        elif char == ord('8'):
            print('ARRIBA\n\r')     
            accion = 0
        elif char == ord('2'):
            print('ABAJO\n\r')   
            accion = 1
        elif char == ord('9'):
            print('ARRIBA+DERECHA\n\r')   
            accion = 4
        elif char == ord('7'):
            print('ARRIBA+IZQUIERDA\n\r')   
            accion = 5
        elif char == ord('3'):
            print('ABAJO+DERECHA\n\r')   
            accion = 6
        elif char == ord('1'):
            print('ABAJO+IZQUIERDA\n\r')   
            accion = 7
                    
        
        print("\rSe va a realizar una acción tipo {}".format(accion))
        
        obs,rew,done,info = env.step(accion)
        print("\rLa recompensa de esta acción ha sido: {0:.3f}".format(rew))
        print("\rOBS toma el valor de [X,Y] = [{},{}]".format(obs['position'][0],obs['position'][1]))
        
        VM = obs['visited_map']
        IM = obs['importance_map']
    
        ax1.imshow(env.render())
        ax2.imshow(VM, cmap = 'gray')
        ax3.imshow(IM,interpolation='nearest', cmap = 'jet_r')
        plt.pause(0.002)
        fig.show()

finally:
    # shut down cleanly
    curses.nocbreak(); screen.keypad(0); curses.echo()
    curses.endwin()
