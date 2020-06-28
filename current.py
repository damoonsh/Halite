from kaggle_environments.envs.halite.helpers import *
import pandas as pd

class Decision_Ship:
    """ 
        Decides ship's next move:

        params: 
            board = the board that we will base our decisions on
            ship = the ship we are deciding for
            step = the steps into the stimulation
        
        returns:
            determine: returns the next-action that should be taken
    """
    # Implement a function that measures the nearest shipyards
    def __init__(self, board, ship):
        # Given values
        self.board = board
        self.step = board.observation['step']
        self.ship = ship
        # Some usefull properties
        self.ship_halite = ship.cell.halite # Ship's halite
        self.player = self.board.current_player  # Player
        # All moves ship can take
        self.moves = {"N": ShipAction.NORTH, 'S': ShipAction.SOUTH, 'W': ShipAction.WEST , 
                      'E' : ShipAction.EAST, 'convert': ShipAction.CONVERT, 'mine': None}
        
        # 5x5 grid around the ship's cell
        self.grid = grid_5(self.ship.cell)
        # Weights of different moves
        self.weights = {"N": 0, "E": 0, "W": 0, "S": 0, "mine": 0, "convert": 0}
        
        self.next_move = None
        
    def determine(self):
        """ Return the next action """
        # Weight different moves:
        
        # First stage: eliminations
        self.first_stage()
        # Get the weights for main four directions
        self.weight_moves()
        # Get the mining weight
        self.weight_mining()
        # get the converting weight
        self.weight_convert()
        
        # Decide between moves
        sorted_weights = {k: v for k, v in sorted(self.weights.items(), key=lambda item: item[1])}
        
        # Choose the action with highest value if it has not been eliminated
        for action, weight in sorted_weights.items():
            if action in self.moves.keys():
                self.next_move = action
        
        return self.next_move

    def weight_mining(self):
        """ Weights mining move for the ship. """
        # Direct correlation with the amount of halite in current_cell - exponential
        # Indirect-corr with the number of steps at the beginning of the game should move around
        self.weights["mine"] = self.ship.cell.halite ** (1.2 + (self.step // 50))


    def weight_convert(self, threshold=2000):
        """ Weights converting for the ship. """
        # Implement: a function that decides if this the best ship to convert to the shipyard
        # Some things to take into account
        # 1. If they are no shipyards left
        no_yards_left = len(self.player.shipyards) == 0
        # 2. If it is the end of the game and we have more than 500 halite in our cargo
        end_of_game_conversion = (self.step > 395 or self.near_end()) and self.ship.halite >= 500
        # 3. There will be a threshold for the amount of cargo any ship could have
        threshhold_reach = self.ship.halite > threshold
        # 4. On shipyard already
        on_shipyard = self.ship.cell.shipyard == None
        
        if (no_yards_left or end_of_game_conversion or threshhold_reach) and not on_shipyard: 
            return 100
        elif not on_shipyard:
            return -5
        else:
            return 2 / (self.step // 50 + 1)
        
    
    
    def weight_moves(self):
        """ This function sets the self.weight parameter for N, W, S, and E. """
        for Dir, cell in self.grid.items():
            # N, W, E, AND S:
            if Dir in self.moves.keys() and len(Dir) == 1: 
                # Instantiate the weight for the Direction
                self.weights[Dir] = self.weight_cell(cell)
                
                # Go through all other ones
                for sub_Dir, sub_cell in self.grid.items():
                    if Dir in sub_Dir and Dir != sub_Dir and len(sub_Dir) == 1:
                        self.weights[Dir] += self.weight_cell(sub_cell)
            
                    
    
    def first_stage(self):
        """
            Eliminate some moves before weighting the moves
            'DONT_GO': avoid that direction
            'GET_away': Don't stay in the current position and took one of the other path than this one
        """
        # If there was a ship/shipyard in E,N,W,or E
        for Dir, cell in self.grid.items():
            if len(Dir) == 1:
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
        # 1. The difference of cell's halite with the current cell's halite
        # 2. If there was a ship then it would be related to the amount of difference in cargo and 1/step
        # 3. If there was my shipyard then it would have a direct corr with the cargo and undirect corr with the steps of the game 
        # 4. If there was an enemy shipyard then + with player's halite, - with # of yards of enemt, - with ship's cargo
        w, cell_ship, cell_shipyard = 0, cell.ship, cell.shipyard
        
        # Mine
        w += (cell.halite - self.ship_halite)
        
        if cell_ship != None:
            myCargo = self.ship.halite
            oppCargo = cell_ship.halite
            # defensive                
            w += (oppCargo - myCargo) * 3 / (self.step // 100 + 1)        

        if cell_shipyard != None:
            if cell_yard.id in self.player.shipyard_ids:
                # Defensive
                w += (self.ship.halite + 5) * (self.step // 20) 
            else:
                oppYards = len(cell_yard.player.shipyards)
                oppHalite = cell_yard.player.halite
                # offensive
                w += (oppHalite // 1000) / (oppYards + 0.5 + self.ship.halite // 10)
        
        return round(w, 3)
    
    
    
    def near_end(self):
        """ Returns True if the game is almost over. """
        count = 0
        # If the halite was less than 500 and it had no ships
        for opp in self.board.opponents:
            if opp.halite < 500 and len(opp.ships) == 0 and self.player.halite > opp.halite: count += 1
        # If count was more than 2 return True
        return count >= 2

    
class ShipLocation:
    def __init__(self, board, ship, grid):
        self.board = ship
        self.ship = ship
        # Get the grid
        self.grid = grid
        
        
    def other_objects(self):
        """ Returns two Dataframes about the information of ships and shipyards on the map. """
        ships_info = {}
        for ship in self.board.ships:
            base_info = {
                "my_ship": False, 
                "halite": 0, "moves": 0
            }
            
            base_info['halite'] = ship.cell.halite
            base_info['moves'] = count_moves(self.ship.position, ship.position)
            
            if ship.id in self.ship.player.ship_ids and ship.id != self.ship.id:
                base_info['my_ship'] = True
                
                ships_info[ship.id] = base_info
            elif ship.id != self.ship.id:
                ships_info[ship.id] = base_info
        
        shipyards_info = {}
        for shipyard in self.board.shipyards:
            base_info = {
                "my_shipyard": False, 
                "halite": 0, "moves": 0
            }
            
            base_info['halite'] = ship.cell.halite
            base_info['moves'] = count_moves(self.ship.position, ship.position)
            
            if shipyard.id in self.ship.player.shipyard_ids:
                base_info['my_shipyard'] = True
                
                shipyards_info[shipyard.id] = base_info
            else:
                shipyards_info[shipyard.id] = base_info
            
        return pd.DataFrame(shipyards_info), pd.DataFrame(ships_info)
        
        
    def generate_grid_df(self):
        """ Generates a Dataframe describing the information of objects and cells in the 5x5 grid of the ship. """
        all_dirs = {}
        
        for direction, cell in self.grid.items():
            
            base_info = {
                "ship_id": None, "shipyard_id": None, 
                "my_ship": False, "my_shipyard": False,
                "halite": 0, "moves": 0
            }
            
            if cell.ship != None: 
                base_info["ship_id"] = cell.ship.id
                if cell.ship.id in self.ship.player.ship_ids:
                    base_info["my_ship"] = True
    
            if cell.shipyard != None: 
                base_info["shipyard_id"] = True
                if cell.shipyard.id in self.ship.player.shipyards_ids:
                    base_info['my_shipyard'] = True
                    
            base_info['halite'] = cell.halite
            # The number of letters in the direction would indicate the number of moves needed to get there
            base_info['moves'] = len(direction)
            
            all_dirs[direcction] = base_info
            
        return pd.DataFrame(all_dirs)
        
####################
# Helper Functions #
####################

def grid_5(cell):
    """
        Returns a dictionary based on the cells in 
        the surrounding 5x5 area of a given cell
    """
    # Main ones
    north, south, west, east = cell.north, cell.south, cell.west, cell.east

    # Secondary ones
    nn, ss, ww,ee = north.north, south.south, west.west, east.east
    
    # The length of the key corresponds to the number of moves needed to get to the cell
    return {
        'N': north, 'S': south, 'W': west, 'E': east, 'NW': north.west, 'NE': north.east, 
        'SW': south.west, 'SE': south.east, 'WW': ww, 'EE': ee, 'NN': nn, 'SS': ss ,
        'NEN': nn.east, 'NWN': nn.west, 'SES': ss.east, 'SWS': ss.west,  'SEE': ee.south, 
        'NEE': ee.north, 'SWW': ww.south, 'NWW': ww.north, 'SEES': ee.south.south , 
        'NEEN': ee.north.north, 'NWWN': ww.north.north, 'SWWS': ww.south.south
    }        


def count_moves(point1, point2, size=21):
    """ 
        Returns the minimum number of between moves 
        to go from point1 to point2.
    """
    # Break the points into coordinates
    x1, y1 = point1.x, point1.y
    x2, y2 = point2.x, point2.y
    
    # For both x and y they are two type of paths to take
    diff_x_1 = abs(x2 - x1) 
    diff_x_2 = abs(size + x2 - x1)
    diff_y_1 = abs(y2 - y1)
    diff_y_2 = abs(size + y2 - y1)
    
    opt1 = diff_x_1 + diff_y_1
    opt2 = diff_x_1 + diff_x_2
    opt3 = diff_x_2 + diff_y_1
    opt4 = diff_x_2 + diff_x_2
    
    return min(opt1, opt2, opt3, opt4)


# Global values
acts = {
    "N": ShipAction.NORTH, 'S': ShipAction.SOUTH,
    'W': ShipAction.WEST , 'E' : ShipAction.EAST,
    'spawn': ShipyardAction.SPAWN, 'convert': ShipAction.CONVERT,
    'mine': None
}

def agent(obs, config):
    # Make the board
    board = Board(obs,config)
    #Step of the board
    step = board.observation['step']
    # Current player info
    me = board.current_player # Player Object
    
    new_board = Board(obs,config)
    
    for ship in me.ships:
        decider = Decision_Ship(new_board, new_board.ships[ship.id], step)
        ship.next_action = decider.determine()
        
        new_board = board.next()
    
    for shipyard in me.shipyards:
        # If there were no ships on the yard
        if new_board.shipyards[shipyard.id].cell.ship == None and step < 392:
            if len(me.ships) == 0:
                shipyard.next_action = acts['spawn']

            if step < 200 and step % 3 == 1:
                shipyard.next_action = acts['spawn']

            if step > 200 and me.halite > 10000 + len(me.ships) * 1000:
                shipyard.next_action = acts['spawn']
        
        new_board = board.next()
        
    return me.next_actions
