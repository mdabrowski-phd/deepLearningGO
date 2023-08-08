#%% Listing 3.16 Your central interface for Go agents
class Agent:
    def __init__(self):
        pass
    
    def select_move(self, game_state):
        raise NotImplementedError()
