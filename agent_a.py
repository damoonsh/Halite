from kaggle_environments.envs.halite.helpers import *

class Decesion_Ship:
    def __init__(self, board, ship, step):
        self.board = board
        self.step = step
        self.ship = ship
        self.ship_halite = ship.cell.halite
        self.keywords = ['GET_AWAY', 'DONT_GO']
        # player
        self.player = self.board.current_player
        self.moves = {"N": ShipAction.NORTH, 'S': ShipAction.SOUTH, 'W': ShipAction.WEST , 
                      'E' : ShipAction.EAST, 'convert': ShipAction.CONVERT,'mine': None}
        
        self.grid = self.grid_5()
        
        
    def determine(self):
        
        if len(self.player.shipyards) == 0:
            return self.moves['convert']
        
        if (self.step >= 395 or self.near_end()) and self.ship.halite > 500:
            return self.moves['convert']

        if self.ship.halite > 1500:
            return self.moves['convert']  
        
        weights = self.weight_moves()
        
        if len(weights) > 0:
            max_move = max(weights, key=weights.get)
        else:
            return self.moves['mine']
        
        if 'mine' in self.moves.keys() and (self.ship.cell.halite) > weights[max_move]:
            return self.moves['mine']
        
        return self.moves[max_move]
        
    
    def weight_moves(self):
        """
            This functions weights different points based on their properties and
            returns a dictionary of weights to choose from.
        """
        # ADD the 1 2 3 4 moves
        # First stage: eliminations
        self.first_stage()
        weights = {}
        
        # Add the weight of all other point to the main four
        for Dir, cell in self.grid["1"].items():
            
            if Dir in self.moves.keys(): 
                # Instantiate the weight for the Direction
                own_weight = self.weight_cell(cell)
                weights[Dir] = own_weight

                # Go through all other ones
                for index in  range(2, 5):
                    for sub_Dir, sub_cell in self.grid[str(index)].items():
                        if Dir in sub_Dir:
                            weights[Dir] += self.weight_cell(sub_cell) 
                            
        return weights
                    
    
    def first_stage(self):
        """
            Eliminate some moves before weighting
            'DONT_GO': avoid that direction
            'GET_away': Dont stay in the current position and took one of the other path than this one
            '': just weight and see which one works better
        """
        # If there was a ship/shipyard in E,N,W,or E
        for Dir, cell in self.grid["1"].items():
            cell_ship = cell.ship
            cell_yard = cell.shipyard
            
            # If there is a ship:
            if cell_ship != None:
                # If it is one of my ships
                if cell_ship.id in self.player.ship_ids:
                    # 'DONT_GO'
                    del self.moves[Dir]
                else:
                    myCargo = self.ship.halite
                    oppCargo = cell_ship.halite
                    # If I had more cargo then get_away
                    if oppCargo < myCargo:
                        # 'GET_AWAY'
                        del self.moves[Dir]
                        # To avoid errors first check to see if the value is there or not
                        if 'mine' in self.moves.keys():
                            del self.moves['mine']
                        
                    
    
    def weight_cell(self, cell):
        """ Weights a cell only based on its properties and relative halite. """
        w = 0
        cell_ship = cell.ship
        cell_yard = cell.shipyard
        
        w += (cell.halite - self.ship_halite) + 2
        
        if cell_ship != None:
            if cell_ship.id in self.player.ship_ids:
                w += cell_ship.halite * -10
            else:
                myCargo = self.ship.halite
                oppCargo = cell_ship.halite
                
                w += (oppCargo - myCargo) * 8
                
        if cell_yard != None:
            if cell_yard.id in self.player.shipyard_ids:
                w += ( self.ship.halite + 10) * 7
            else:
                oppYards = len(cell_yard.player.shipyards)
                w += 1 / (oppYards + 1) * 10
        
        return round(w, 3)
                    
    def near_end(self):
        """ Returns True if the game is almost over. """
        count = 0
        # If the halite was less than 500 and it had no ships
        for opp in self.board.opponents:
            if opp.halite < 500 and len(opp.ships) == 0 and self.player.halite > opp.halite: count += 1
        # If count was more than 2 return True
        return count >= 2
    
    def grid_5(self):
        """
            Returns a 5x5 dictionary as a dictionary
            1 means cells where you can reach with one move, etc.
        """
        north = self.ship.cell.north
        south = self.ship.cell.south
        west = self.ship.cell.west
        east = self.ship.cell.east
        
        nn = north.north
        ss = south.south
        ww = west.west
        ee = east.east
        
        return {
            '1': {'N': north, 'S': south, 'W': west, 'E': east}, 
            '2': {
                    'NW': north.west, 'NE': north.east, 'SW': south.west, 'SE': south.east,
                    'WW': ww, 'EE': ee, 'NN': nn, 'SS': ss
                },
            '3': {
                 'NEN': nn.east, 'NWN': nn.west, 'SES': ss.east, 'SWS': ss.west, 
                 'SEE': ee.south, 'NEE': ee.north, 'SWW': ww.south, 'NWW': ww.north
             },
            '4': {'SEES': ee.south.south , 'NEEN': ee.north.north, 'NWWN': ww.north.north, 'SWWS': ww.south.south}
        }
        

def get_neighbors(cell):
    # returns sorounding cells for a point
    return {
        'N': cell.north, 'NW': cell.north.west, 'NE': cell.north.east,
        'S': cell.south, 'SW': cell.south.west, 'SE': cell.south.east,
        'WW': cell.west.west, 'EE': cell.east.east,
        'NN': cell.north.north, 'SS': cell.south.south,
        'W': cell.west, 'E': cell.east
    }

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


def choose_between(l=[]):
    import random
    
    if l != []:
        random.choice(l)
    
    return random.choice([acts['N'], acts['W'],acts['E'],acts['S'], acts['mine']])

def log(text, step=1):
    if step == 0:
        with open("log-a.txt", "w") as text_file:
            text = str(text) + '\n'
            text_file.write(text)
    else:
        with open("log-a.txt", "a") as text_file:
            text = str(text) + '\n'
            text_file.write(text)

log('logs:', 1)

def agent(obs, config):
    # Make the board
    board = Board(obs,config)
    #Step of the baord
    step = board.observation['step']
    # Current player info
    me = board.current_player # Player Object
    
    new_board = Board(obs,config)
    log('-----------------------------------------------------------------')
    log(step + 1)
    for ship in me.ships:
        log('ship-id:' + ship.id + ', pos:' + str(ship.position) + ', cargo: ' + str(ship.halite))
#         decider = Decesion_Ship(new_board, ship, step)
        decider = Decesion_Ship(new_board, new_board.ships[ship.id], step)
        ship.next_action = decider.determine()
        
        new_board = board.next()
    
    #Implemenet a pipeline where given that
    for shipyard in me.shipyards:
        # If there were no ships on the yard
        if new_board.shipyards[shipyard.id].cell.ship == None and step < 392:
            if len(me.ships) == 0:
                shipyard.next_action = acts['spawn']

            if step < 150 and step % 3 == 1:
                shipyard.next_action = acts['spawn']

            if step > 200 and me.halite > 10000 + len(me.ships) * 1000:
                shipyard.next_action = acts['spawn']
        
        new_board = board.next()

        
    return me.next_actions
