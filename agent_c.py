
from kaggle_environments.envs.halite.helpers import *

def get_neighbors(cell):
    # returns sorounding cells for a point
    return {
        'N': cell.north, 'NW': cell.north.west, 'NE': cell.north.east,
        'S': cell.south, 'SW': cell.south.west, 'SE': cell.south.east,
        'WW': cell.west.west, 'EE': cell.east.east,
        'NN': cell.north.north, 'SS': cell.south.south,
        'W': cell.west, 'E': cell.east
    }

def weight_cell(cell, step, cargo, player_halite, yard_ids, ship_ids, ships):
    cell_ship = cell.ship
    cell_yard = cell.shipyard
    halite = cell.halite
    
    w_ship, w_yard, w_halite = 0, 0, 0
    
    # If there was a ship
    if cell_ship != None:
        # If the ship was mine:
        if cell_ship.id in ship_ids:
            w_ship = -10000
        else:
            if cell_ship.halite > cargo:
                w_ship = 1000
            else:
                w_ship = -1000
                w_halite = -10
                
    # If there was yard
    if cell_yard != None:
        # If it was mine:
        if cell_yard.id in yard_ids:
            w_yard = cargo ** 2 / 1000
        else:
            w_yard = 1 / (cargo + 1) ** (1 / 3)
            w_halite = -10
    
    # If w_halite was not determined by other entities
    if w_halite == 0:
        w_halite = (step ** 2 * (halite - 5) ** 3) / (player_halite + 10) ** (1/3)
            
    return w_halite + 20 * w_ship * 10 + w_yard

def heat_map(ship, step):
    cargo = ship.halite
    cell = ship.cell
    
    neighbors = get_neighbors(cell)

def basic_moves(cell):
    neighbors = get_neighbors(cell)
    
    moves = ['N', "W", 'E', 'W']
    
    for key, cell in directions.items():
        if cell.ship == None or cell.shipyard == None and key in moves:
            moves.remove(key)
    
    return moves

def check_cell_prop(cell, ship_ids, yard_ids, cargo, player_halite):
    """
         0: means don't go in this direction but don't need to avoid it
         1: means go in this direction
        -2: means don't in this direction.
        -1: means get away with any of other direction
    """
    if cell.ship != None: # If there was a ship
        if cell.ship.id in ship_ids:
            # Don't go in this direction
            return -2
        else:
            if cell.ship.halite < cargo:
                return 1 * cell.ship.halite ** 2
            else:
                return -1
            
    elif cell.shipyard != None: # If there was a shipyard
        if cell.shipyard.id in yard_ids:
            # If it was mine
            if cargo > 500:
                return 1 * cargo ** 2
            else:
                return 0.5 * cargo
        else:
            if cell.shipyard.player.halite > player_halite and cargo < 1000:
                return 1 * (cell.shipyard.player.halite - player_halite)
            else:
                # If there was an enemy shipyard and we had a lot of cargo don't attack
                return -2
    else:
        return 0
    

acts = {
    "N": ShipAction.NORTH, 'S': ShipAction.SOUTH,
    'W': ShipAction.WEST , 'E' : ShipAction.EAST,
    'spawn': ShipyardAction.SPAWN, 'convert': ShipAction.CONVERT,
    'mine': None
}

def randomize(choices=[]):
    import random
    
    if len(choices) == 0:
        choices = list(acts.items())
    
    choice = random.choice(choices)
    
    if choice[0] == 'spawn' or  choice[0] == 'convert':
        return randomize(exclude)
    
    return choice[1]


def choose_between(choices):
    import random
    
    return random.choice(choices)

awaited_actions = {}

def agent(obs, config):
    # Make the board
    board = Board(obs,config)
    #Step of the baord
    step = board.observation['step']
    # Current player info
    me = board.current_player # Player Object
    me_id = board.current_player_id # Player ID
    ships_ids = me.ship_ids
    shipyard_ids = me.shipyard_ids
    ships = me.ships # Ships list
    shipyards = me.shipyards # Yards list
    player_halite = me.halite # Player's halite
    
    # Get the  ship_ids and retrieve them each time from the board
    # Decesion for ships
    for Id in ships_ids:
        # Get each ship from the board itself since after each update 
        # the ships won't know their relative positions
        # Get the ship's info
        ship = board.ships[Id]
        
        ship_cell = ship.cell
        ship_cell_halite = ship_cell.halite
        cargo = ship.halite
        
        moves = {'N': ship_cell.north, 'S': ship_cell.south, 'W': ship_cell.west, 'E': ship_cell.east}        
        nominees = ['N', 'S', 'W', 'E']
        mine = True
        declared = False
        
        if (cargo > 1000 + step ** (step / (cargo + 10))) or step == 399:
            ship.next_action = acts['convert']
            declared = True
            mine = False
        
        if ship_cell.shipyard != None and not declared: 
            mine = False
        elif ship_cell_halite < 30:
            mine = False
        
        # Mine if there is no ships in E, N, S, and W with less cargo
        for Dir, cell in moves.items():
            if check_cell_prop(cell, me.ship_ids, me.shipyard_ids, cargo, player_halite) == -1:
                nominees.remove(Dir)
                mine = False
            elif check_cell_prop(cell, me.ship_ids, me.shipyard_ids, cargo, player_halite) == -2:
                nominees.remove(Dir)
                
                
        if mine:
            if step == 399:
                ship.next_action = acts['convert']
                declared = True
            else:
                ship.next_action = acts['mine']
                declared = True
                
        elif not declared:
            val = -1e8
            best_move = []
            
            for move in nominees:
                if check_cell_prop(moves[move], me.ship_ids, me.shipyard_ids, cargo, player_halite) >= val:
                    val = check_cell_prop(moves[move], me.ship_ids, me.shipyard_ids, cargo, player_halite)
                    best_move.append(move)
            
            if len(best_move) != 1 :
                # If they were ties
                ship.next_action = acts[choose_between(nominees)]
                declared = True
            elif len(best_move) == 0:
                ship.next_action = best_move[0]
                declared = True 
        
        if not mine and len(nominees) == 0: 
            declared = True
            ship.next_action = acts['mine']
        
        if not declared: ship.next_action = acts[choose_between(nominees)]
        
        # Update the board after each action for ships
        board = board.next()
    
    # Decesion for shipyards
    for Id in shipyard_ids:
        yard = board.shipyards[Id]
        
        if step < 9:
            yard.next_action = acts['spawn']
            
            board = board.next()
    
        elif len(ships) < 3 and player_halite > 3500 and step < 398:
            
            yard.next_action = acts['spawn']
            
            board = board.next()
    
#     if step < 7 and len(shipyards) != 0:
#             shipyards[0].next_action = acts['spawn']
    
    # If there are no ships, use first shipyard to spawn a ship.
    if len(ships) == 0 and len(shipyards) > 0 and step != 399:
        shipyards[0].next_action = acts['spawn'] 
        
    # If there are no shipyards, convert first ship into shipyard.
    if len(shipyards) == 0 and len(ships) > 0:
        ships[len(ships) // 2].next_action = acts['convert'] 
        
    return me.next_actions
