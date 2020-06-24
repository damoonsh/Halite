from kaggle_environments.envs.halite.helpers import *

class Decesion_Ship:
    """ 
        Decides ship's next move:
        params: 
            board = the board that we will base our decisions on
            ship = the ship we are deciding for
            step = the steps into the situmalation
    """
    # Implement a function that measures the nearest shipayrds
    def __init__(self, board, ship, step):
        # Given values
        self.board = board
        self.step = step
        self.ship = ship
        # Some usefull properties
        self.ship_halite = ship.cell.halite # Ship's halite
        self.player = self.board.current_player  # Player
        # All moves ship can take
        self.moves = {"N": ShipAction.NORTH, 'S': ShipAction.SOUTH, 'W': ShipAction.WEST , 
                      'E' : ShipAction.EAST, 'convert': ShipAction.CONVERT,'mine': None}
        # 5x5 grid around the ship's cell
        self.grid = grid_5(self.ship.cell)
        
        
    def determine(self):
        """ Return the next action """
        ##################################################################################
        # This parts should be implemented differently
        # If they were no yards, then convert a ship
        if len(self.player.shipyards) == 0:
            # Implement: find the best ship in the board that could trun in to a yard
            return self.moves['convert'] 
        
        # Given that steps is more than a certain amount convert ship with halite more than 500
        # Sometimes the trial does not last 393 and I should look for the number of different 
        # agent's halite and ships and yards
        if (self.step > 393 or self.near_end()) and self.ship.halite >= 500:
            return self.moves['convert'] 
        
        # If a ship has more than 2500 halite then make it a yard
        if self.ship.halite > 2500:
            return self.moves['convert']  
        ##################################################################################
        
        # Implement: more robust way of comparing all the possible moves
        
        # Weight different moves
        weights = self.weight_moves()
        
        # Check to see if not all of the moves were deleted from dictionary
        if len(weights) > 0:
            max_move = max(weights, key=weights.get)
        else:
            # If all were deleted  then mine
            return self.moves['mine']
        
        ## Improve this part
        # If the weights were not high enough
        if 'mine' in self.moves.keys() and (self.ship.cell.halite) > weights[max_move]:
            return self.moves['mine']
        
        return self.moves[max_move]
        
    
    def weight_moves(self):
        """
            This functions weights different points based on their properties and
            returns a dictionary of weights to choose from.
        """
        # First stage: eliminations
        self.first_stage()
        weights = {}
        
        # Add the all other point to the main four with their corresponding weights
        for Dir, cell in self.grid[1].items():
            
            if Dir in self.moves.keys(): 
                # Instantiate the weight for the Direction
                weights[Dir] = self.weight_cell(cell)
                
                # Each cell will be multiplied by a weight given that it takes 
                # different number of steps to get to that point
                move_weight = {2: 0.8, 3: 0.7, 4: 0.5}
                
                # Go through all other ones
                for index in  range(2, 5):
                    for sub_Dir, sub_cell in self.grid[index].items():
                        if Dir in sub_Dir:
                            weights[Dir] += self.weight_cell(sub_cell) * move_weight[index]
                            
        return weights
                    
    
    def first_stage(self):
        """
            Eliminate some moves before weighting
            'DONT_GO': avoid that direction
            'GET_away': Dont stay in the current position and took one of the other path than this one
            '': just weight and see which one works better
        """
        
        # If there was a ship/shipyard in E,N,W,or E
        for Dir, cell in self.grid[1].items():
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
        # Change: the random weights in this block should be changed with tendencies
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

    

class Decesion_shipyard:
    """ Decides a move for the shipyard. """
    def __init__(self, board, shipyard, step):
        # Setting the values
        self.yard = shipyard
        self.board = board
        self.step = step
        # Possible moves
        self.moves = {'convert': ShipyardAction.SPAWN,'stay': None}
        self.grid = grid_5(shipyard.cell)
        
    def determine():
        """ Returns the desirebale action. """
        pass


    
class LocateObject:
    """ 
        Gets a board and a object (ship/shipyard) and returns a dictionary containing
        the relative information of other objects to the passed object 
    """
    def __init__(self, board, obj):
        self.board = board
        self.player = obj.player
        self.obj = obj
        self.pos = obj.position
        # Get the current_player's ship and shipyard ids
        self.ship_ids = obj.player.ship_ids
        self.shipyard_ids = obj.player.shipyard_ids
    
    def souroundings():
        """ Returns a dict containing the list of objects nearby the given object. """
        my_ships, opp_ships = self.locate_ships()
        my_yards, opp_ships = self.locate_yards()
        
        return {
            "my_ships": my_ships, "opp_ships": opp_ships, "my_yards": my_yards, "opp_ships": opp_ships
        }
    
    
    def locate_yards(self):
        # Get the enemy and player ships
        my_yards_dict, opp_yards_dict = {}, {}
        
        for shipyard in self.board.shipyards:
            if shipyard.id != self.obj.id  and shipyard in self.player.shipyards:
                my_yards_dict[shipyard.id] = count_moves(self.pos, shipyard.position)
            elif shipyard.id != self.obj.id:
                opp_yards_dict[shipyard.id] = count_moves(self.pos, shipyard.position)
            
        return my_yards_dict, opp_yards_dict
    
    
    def locate_ships(self):
        # Get the enemy and player ships
        my_ships_dict, opp_ships_dict = {}, {}
        
        for ship in self.board.ships:
            if ship.id != self.obj.id and ship in self.player.ships:
                my_ships_dict[ship.id] = count_moves(self.pos, ship.position)
            elif ship.id != self.obj.id:
                opp_ships_dict[ship.id] = count_moves(self.pos, ship.position)
            
        return my_ships_dict, opp_ships_dict
    
    
class ShipTendency:
    """ 
        Weights different options for either to be offensive or deffensive for
        a given ship at any position on the board
    """
    # 1. Look for yards where they might be in danger:
    #    Check the souroundins of the yards to make sure they are not threatened
    # Note: generally speaking go over all of the 
    def __init__(self, board, ship):
        # Get the values
        self.board = board
        self.ship = ship
        self.cargo = ship.halite
        # Constructiong a grid
        self.grid = grid_5(cell)
        # The distance of all objects in the board relative to our ship
        self.objects = LocateObject(board, ship).souroundings()
    

class ShipyardTendency:
    """ 
        Weights different options for either to be offensive or deffensive for
        a given shipyard at any position on the board
    """
    def __init__(self, board, shipyard):
        # Get the values
        self.board = board
        self.yard = shipyard

        
###############################
# Helper Functions for objects#
###############################
def grid_5(cell):
    """
        Returns a dictionary based on the cells in 
        the surrounding 5x5 area of a given cell
    """
    # Main ones
    north = cell.north
    south =cell.south
    west = cell.west
    east = cell.east
    # Secondary ones
    nn = north.north
    ss = south.south
    ww = west.west
    ee = east.east
        
    return {
        1: {'N': north, 'S': south, 'W': west, 'E': east}, 
        2: {
            'NW': north.west, 'NE': north.east, 'SW': south.west, 'SE': south.east,
            'WW': ww, 'EE': ee, 'NN': nn, 'SS': ss
            },
        3: {
             'NEN': nn.east, 'NWN': nn.west, 'SES': ss.east, 'SWS': ss.west, 
             'SEE': ee.south, 'NEE': ee.north, 'SWW': ww.south, 'NWW': ww.north
        },
        4: {'SEES': ee.south.south , 'NEEN': ee.north.north, 'NWWN': ww.north.north, 'SWWS': ww.south.south}
    }        


def sigmoid(val):
    """ Given a value, feeds it to a sigmoid function. """
    import numpy as np
    
    return 1 / 1 + np.exp(-1 * val)



def count_moves(point1, point2, size=21):
    """ 
        Returns the minimum number of between moves 
        to go from point1 to point2. 
        Based on the negativity of diff_x and diff_y,, we can decide the direction
        {[id]: {'num': int, 'diff_x': int, 'diff_y': int}, ...}
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
    #Step of the baord
    step = board.observation['step']
    # Current player info
    me = board.current_player # Player Object
    
    new_board = Board(obs,config)
    
    for ship in me.ships:
        
        decider = Decesion_Ship(new_board, new_board.ships[ship.id], step)
        ship.next_action = decider.determine()
        
        new_board = board.next()
    
    #Implemenet a pipeline where given that
    for shipyard in me.shipyards:
        # If there were no ships on the yard
        if new_board.shipyards[shipyard.id].cell.ship == None and step < 392:
            if len(me.ships) == 0:
                shipyard.next_action = acts['spawn']

            if step < 100 and step % 3 == 1:
                shipyard.next_action = acts['spawn']

            if step > 200 and me.halite > 10000 + len(me.ships) * 1000:
                shipyard.next_action = acts['spawn']
        
        new_board = board.next()
        
    return me.next_actions
