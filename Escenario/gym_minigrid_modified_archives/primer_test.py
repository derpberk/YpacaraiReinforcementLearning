import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

def redraw(img):
    
    img = env.render('rgb_array')

    window.show_img(img)

def reset():
    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

def step(action):
    obs, reward, done, info = env.step(action)
    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        print('done!')
        reset()
    else:
        redraw(obs)

    

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    elif event.key == 'backspace':
        reset()

    elif event.key == 'left':
        step(env.actions.left)
    elif event.key == 'right':
        step(env.actions.right)
    elif event.key == 'up':
        step(env.actions.forward)
    elif event.key == 'down':
        step(env.actions.backward)

    # Spacebar
    elif event.key == 'w':
        step(env.actions.right_up)
    elif event.key == 'a':
        step(env.actions.right_down)
    elif event.key == ' ':
        step(env.actions.left_up)
    elif event.key == 'd':
        step(env.actions.left_down)
    
    # print("Estado observado:")
    # print(obs)
    
    return

# Cargamos el entorno #

env = gym.make('MiniGrid-Playground-v1')

# Cargamos la misma semilla siempre#
#env = ReseedWrapper(env)

# Definimos que la observacion del entorno será de tipo IMAGEN RGB #
#env = RGBImgObsWrapper(env)
# Para hacer que la observacion sea el entorno entero #
env = FullyObsWrapper(env)

window = Window('gym_minigrid - Test 1')
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)