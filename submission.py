
from kaggle_environments.envs.halite.helpers import *

def agent(obs,config):
    print(dir(obs))
    board = Board(obs,config)
    me = board.current_player
    
    # Set actions for each ship
    for ship in me.ships:
        ship.next_action = ShipAction.NORTH
    
    # Set actions for each shipyard
    for shipyard in me.shipyards:
        shipyard.next_action = None
    
    return me.next_actions
