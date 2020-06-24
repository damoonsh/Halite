from kaggle_environments.envs.halite.helpers import *

    
class LocateObject:
    """ 
       params:
            board: board to play on
            obj: either a shipyard/ship 
    """
    def __init__(self, board, obj):
        self.board = board
        self.player = obj.player
        self.obj = obj
        self.pos = obj.position
        # Get the current_player's ship and shipyard ids
        self.ship_ids = obj.player.ship_ids
        self.shipyard_ids = obj.player.shipyard_ids
    
    def locate_yards(self):
        """ 
            Get the number of moves each shipyard is from our 
            object and set it in a dictionaries 
        """
        # Get the enemy and player ships
        my_yards_dict, opp_yards_dict = {}, {}
        
        for shipyard in self.board.shipyards:
            if shipyard.id != self.obj.id  and shipyard in self.player.shipyards:
                my_yards_dict[shipyard.id] = count_moves(self.pos, shipyard.position)
            elif shipyard.id != self.obj.id:
                opp_yards_dict[shipyard.id] = count_moves(self.pos, shipyard.position)
            
        return my_yards_dict, opp_yards_dict
    
    
    def locate_ships(self):
        """ 
            Get the number of moves each ship is from our 
            object and set it in a dictionaries 
        """
        # Get the enemy and player ships
        my_ships_dict, opp_ships_dict = {}, {}
        
        for ship in self.board.ships:
            if ship.id != self.obj.id and ship in self.player.ships:
                my_ships_dict[ship.id] = count_moves(self.pos, ship.position)
            elif ship.id != self.obj.id:
                opp_ships_dict[ship.id] = count_moves(self.pos, ship.position)
            
        return my_ships_dict, opp_ships_dict

    
    def surroundings(self):
        """ Returns a dict containing the list of objects nearby the given object. """
        my_ships, opp_ships = self.locate_ships()
        my_yards, opp_ships = self.locate_yards()
        
        return {
            "my_ships": my_ships, "opp_ships": opp_ships, "my_yards": my_yards, "opp_ships": opp_ships
        }

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

        self.hyperparameters = ShipTendency(board, ship).tend()
        
        
    def determine(self):
        """ Return the next action """
        # Implement: Make on all the decisions solely on weights
        ##################################################################################
        # This parts should be implemented differently
        # If they were no yards, then convert a ship
        if len(self.player.shipyards) == 0:
            # Implement: find the best ship in the board that could turn in to a yard
            return self.moves['convert'] 
        
        # Given that steps is more than a certain amount convert ship with halite more than 500
        # Sometimes the trial does not last 393 and I should look for the number of different 
        # agent's halite and ships and yards
        if (self.step > 395 or self.near_end()) and self.ship.halite >= 500:
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
        
    
    def weight_mining(self):
        pass


    def weight_convert(self):
        pass
    
    
    def weight_moves(self):
        """
            This functions weights different points based on their properties and
            returns a dictionary of weights to choose from.
        """
        # First stage: eliminations
        self.first_stage()
        weights = {}
        
        # Add the all other point to the main four with their corresponding weights
        for Dir, cell in self.grid.items():
            
            if Dir in self.moves.keys() and len(Dir) == 1: 
                # Instantiate the weight for the Direction
                weights[Dir] = self.weight_cell(cell)
                
                # Each cell will be multiplied by a weight given that it takes 
                # different number of steps to get to that point
                move_weight = {2: 0.8, 3: 0.7, 4: 0.65}
                
                # Go through all other ones
                for sub_Dir, sub_cell in self.grid.items():
                    if Dir in sub_Dir and Dir != sub_Dir and len(sub_Dir) == 1:
                        weights[Dir] += self.weight_cell(sub_cell) * move_weight[len(sub_cell)]
                            
        return weights
                    
    
    def first_stage(self):
        """
            Eliminate some moves before weighting the moves
            'DONT_GO': avoid that direction
            'GET_away': Don't stay in the current position and took one of the other path than this one
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
        
        # Mine
        w += (cell.halite - self.ship_halite) + 2
        
        if cell_ship != None:
            if cell_ship.id in self.player.ship_ids:
                # Avoid clash
                w += cell_ship.halite * -10
            else:
                myCargo = self.ship.halite
                oppCargo = cell_ship.halite
                # defensive                
                w += (oppCargo - myCargo) * self.hyperparameters['offensive']
                
        if cell_yard != None:
            if cell_yard.id in self.player.shipyard_ids:
                # Defensive
                w += ( self.ship.halite + 10) * self.hyperparameters['defensive']
            else:
                oppYards = len(cell_yard.player.shipyards)
                # offensive
                w += 1 / (oppYards + 1) * self.hyperparameters['offensive']
        
        return round(w, 3)
    
    
    def near_end(self):
        """ Returns True if the game is almost over. """
        count = 0
        # If the halite was less than 500 and it had no ships
        for opp in self.board.opponents:
            if opp.halite < 500 and len(opp.ships) == 0 and self.player.halite > opp.halite: count += 1
        # If count was more than 2 return True
        return count >= 2

    

class Decision_shipyard:
    """ Decides a move for the shipyard. """
    def __init__(self, board, shipyard, step):
        # Setting the values
        self.yard = shipyard
        self.board = board
        self.step = step
        # Possible moves
        self.moves = {'convert': ShipyardAction.SPAWN, 'stay': None}
        self.grid = grid_5(shipyard.cell)
        
    def determine(self):
        """ Returns the desirebale action. """
        pass
    
class ShipTendency:
    """ 
        Given the ship's situation and properties, weights different 
        tendencies and returns a set of relatively scaled weights.
        This module operates as a hyper parametert producer 
        for any given Ship.

        params:
            board: Board that events are occuring
            ship: Our ship
    """
    def __init__(self, board, ship):
        # Get the values
        self.board = board
        self.ship = ship
        self.cargo = ship.halite

        # Construct a grid
        self.grid = grid_5(ship.cell)
        # Get the stat of the grid that ship is in
        self.grid_stat = get_grid_stat(self.grid)
        
        # The distance of all objects in the board relative to our ship
        self.objects = LocateObject(board, ship).surroundings()

        # Initiate the values that are going to be returned 
        self.defensive = 0
        self.offensive = 0
        self.mine = 0


    def base_analysis(self):
        """ Instantiate the initial values for different tendencies. """
        
        # If the ship did not have any of it's own yards in it's grid
        if len(self.grid_stat['my_shipyards']) == 0: 
            self.defensive += 10
        else:
            self.defensive += 1
            self.offensive += 5
            self.mine += 2

        # If there were more enemy ships in the area than my own ships
        if len(self.grid_stat['my_ships']) <= len(self.grid_stat['opp_ships']):
            self.defensive += 10
        else:
            self.offensive += 6
            self.mine += 4
        
        # If there were more enemy shipyards in the area than my own shipyards
        if len(self.grid_stat['my_shipyards']) <= len(self.grid_stat['opp_shipyards']):
            self.defensive += 12
            self.offensive += 5
        else:
            self.defensive += 5
            self.offensive += 8
            self.mine += 10


    def analyze_yard_stat(self):
        """ 
            This function will focus on evaluating the self.defensive weight variable. 
            Note: the analysis is solely based on the distances
        """
        # Implement: Go through my yards and determine if they need 
        # protection by adding to the self.deffensice value
        for shipyard in self.objects['my_yards']:
            # Get the objects around shipyard
            objects_around = LocateObject(board, shipyard).surroundings()
            
            # Evaluate defensive with respect to the distance of enemy ships
            for Id, distance in objects_around['opp_ships'].items():
                # Less distances means that we need protection
                self.defensive += 2 / abs(distance + 1)

            # Evaluate defensive and offensive with respect to the distance of my ships
            for Id, distance in objects_around['my_ships'].items():
                self.offensive += 1.5 / abs(distance + 1)
        

    def analyze_ship_stat(self):
        """ 
            This function will focus on evaluating the self.offensive weight variable. 

            Note: 
                Although the analysis is solely based on the distances of 
                objects yet it is important to take into account that the 
                difference in cargo for more accurate analysis.
        """
        for ship in self.objects['my_ships']:
            # Get the objects around shipyard
            objects_around = LocateObject(board, ship).surroundings()

            # If I have my ships around I can be offensive
            for Id, distance in objects_around['my_ships'].items():
                self.offensive += 2 / abs(distance + 1)

            # If there are enemy ships around then I should be more defensive
            for Id, distance in objects_around['opp_ships'].items():
                self.defensive += 2 / abs(distance + 1)

            # If there are enemy ships around then I should be more defensive
            for Id, distance in objects_around['opp_yards'].items():
                self.offensive += 2 / abs(distance + 1)

            # If there are my shipyards around then I can be more offensive than defensive
            for Id, distance in objects_around['my_yards'].items():
                self.offensive += 2 / abs(distance + 1)
                self.defensive += 1 / abs(distance + 1)

   
    def scale(self):
        """ Changes the weights into values between 0 and 1. """
        Sum = self.defensive + self.offensive + self.mine # Sum of all the weights
        # Scaling:
        self.defensive /= Sum
        self.offensive /= Sum
        self.mine /= Sum         
            

    def tend(self):
        # Do the analysis
        self.base_analysis()
        self.analyze_ship_stat()
        self.analyze_yard_stat()

        # Scale the weights
        self.scale()

        # Note: the returned values are considered to be hyperparameters of the
        # actual weightings and are positive values between 0 and one
        return {
            'defensive': self.defensive,
            'offensive': self.offensive,
            'mine': self.mine
        }


class ShipyardTendency:
    """ 
        Weights different options for either to be offensive or defensive for
        a given shipyard at any position on the board
    """
    def __init__(self, board, shipyard):
        # Get the values
        self.board = board
        self.yard = shipyard

        
####################
# Helper Functions #
####################
def get_grid_stat(grid, player):
    """ Returns a dictionary containing objects in a grid. """
    stat = {'my_ships': [], 'my_shipyards': [], 'opp_ships': [], 'opp_shipyards': [], 'NoObject': []}
    
    ship_ids = player.ship_ids
    shipyard_ids = player.shipyard_ids

    for Dir, cell in grid.items():    
        if grid[Dir].ship != None:
            if grid[Dir].ship.id in ship_ids:
                stat['my_ships'].append(grid[Dir].ship.id)
            else:
                stat['opp_ships'].append(grid[Dir].ship.id)

        elif grid[Dir].shipyard != None:

            if grid[Dir].shipyard.id in shipyard_ids:
                stat['my_shipyards'].append(grid[Dir].shipyard)
            else:
                stat['opp_shipyards'].append(grid[Dir].shipyard)
        else:
            stat['NoObject'].append(cell)

    return stat


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

            if step < 100 and step % 3 == 1:
                shipyard.next_action = acts['spawn']

            if step > 200 and me.halite > 10000 + len(me.ships) * 1000:
                shipyard.next_action = acts['spawn']
        
        new_board = board.next()
        
    return me.next_actions
