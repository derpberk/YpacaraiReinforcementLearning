from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from numpy import genfromtxt

class PlaygroundV1(MiniGridEnv):
    """
    Environment created for Paraguayan issues ;)
    """

    def __init__(self):

        grid_map = genfromtxt('/home/samuel/MachineLearning/Entornos/Ypacarai/YpacaraiMap.csv', delimiter=',',dtype = int)

        agent_start_pos = (7,10) # Posicion inicial, por ejemplo #
        agent_start_dir = 3 # Por ejemplo #

        size = grid_map.shape # Sacamos las dimensiones
        width = size[1]
        height = size[0]
        
        self.height = height
        self.width = width

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        

        # Creamos un mapa de visitacion #
        self.visited_map = np.zeros((width,height))

        # Lo inicializamos con el punto inicial #

        self.visited_map[agent_start_pos] = 1 # El punto inicial se visita #
        
        # Creamos la matriz de recompensa dinámica (mapa de calor)#
        
        self.R_dyn = np.ones((width,height))*10
        self.R_dyn[agent_start_pos] = 0 # El lugar de partida no tiene reward
        self.agent_pos_ant = agent_start_pos # Inicializamos la posicion anterior
        

        super().__init__(
            width=width,
            height=height,
            max_steps=1000000,
            agent_view_size = 1,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):

        # Creamos el grid #
        self.grid = Grid(width, height)
        grid_map = genfromtxt('/home/samuel/MachineLearning/Entornos/Ypacarai/YpacaraiMap.csv', delimiter=',',dtype = int)
        
        # Definimos los objetos de tierra y agua con los colores adecuados #

        tierra = Wall('green') # 
        agua = Floor('blue') #

        # Rellenamos el entorno con el layout del map_grid #
        for i in range(width):
            for j in range(height):

                if(grid_map[j][i] == 0):
                    self.grid.set(i,j,tierra)
                else:
                    self.grid.set(i,j,agua)
                    
        self.grid.set(self.agent_start_pos[0],self.agent_start_pos[1],Floor('red'))            

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # No explicit mission in this environment
        self.mission = 'Mision: Disfrutar del Lago Ypacarai'

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        return obs, reward, done, info
    
    # Funcion de actualizacion del mapa R de calor #
    def actualiza_R(self):
        
        for i in range(self.width):
            for j in range(self.height):
                
                if self.agent_pos[0] == i and self.agent_pos[1] == j:
                    self.R_dyn[i][j] = 0
                    self.grid.set(i,j,Floor('red'))
                else:
                    self.R_dyn[i][j] = np.min([self.R_dyn[i][j]+0.5,10])
                

register(
    id='MiniGrid-Playground-v1',
    entry_point='gym_minigrid.envs:PlaygroundV1'
)
