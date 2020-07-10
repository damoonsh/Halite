from kaggle_environments.envs.halite.helpers import *
import pandas as pd

class Decision_Ship:
    """ 
        Decides ship's next move:
        params:
            board: the board that we will base our decisions on
            ship: the ship we are deciding for
            step: the steps into the stimulation
        returns:
            determine: returns the next-action that should be taken
    """
    def __init__(self, board, ship):
        # Given values
        self.board = board
        self.step = board.observation['step']
        self.ship = ship
        self.ship_cargo = ship.halite
        self.current_cell = ship.cell
        self.current_position = ship.position
        
        # Some usefull properties
        self.player = self.board.current_player  # Player
        self.simple_moves = {'N': ship.cell.north, 'S': ship.cell.south, 'W': ship.cell.west, 'E': ship.cell.east}
        
        # All moves ship can take
        self.moves = {"N": ShipAction.NORTH, 'S': ShipAction.SOUTH, 'W': ShipAction.WEST, 
                      'E' : ShipAction.EAST, 'convert': ShipAction.CONVERT, 'mine': None}
    
        # Weights of different moves
        self.weights = {"N": 0, "E": 0, "W": 0, "S": 0, "mine": 0, "convert": 0}
        
        # The object's relative situation to other ship/shipyards
        self.locator = Locator(board, ship)
        self.Ships = self.locator.get_ship_info()
        self.Shipyards = self.locator.get_shipyard_info()
        self.grid = self.locator.generate_grid_df()
        
        # Default move which is set to mining (None)
        self.next_move = None
        
        
    def determine(self):
        """ Returns the next action decided for the ship based on the observations that have been made. """
        # Get the weights for main four directions
        self.weight_moves()
        
        # Decide between moves
        sorted_weights = {k: v for k, v in sorted(self.weights.items(), key=lambda item: item[1], reverse=True)}
        
        log('  weights:' + str(sorted_weights))
        
        # Choose the action with highest value if it has not been eliminated
        for action in sorted_weights.keys():
            if action in self.moves.keys():
                return self.moves[action]
        
        # If none were chosen, then just return the default move which is mining
        return self.next_move


    def weight_moves(self):
        """ 
            This is the main function and runs other helper functions within the module to in order to weight
            the different moves that are going to be taken.

            The weightings will take numerous considerations into account (halite, closeness to shipyard, etc.)
            yet there are going to be multiplied by some values manually in order to encourage or discourage certain
            moves. In order to keep things simple, the values will be multiplied by a number between -10 to 10.

            Now some cells are closer to the current cell and in order to put an emphasis on that, we will take into 
            account the fact that distance would have a indirect correlation in our weighting process.

            When going through the different directions on the ship's grid, they are going to be some scenarios that
            we should  decide for them accordingly:
                1. If there is one of my ships: should avoid collision
                2. If there is an enemy shipyard: depending on variables, could either runaway or attack it
                3. If there is one of my shipyards: possibly deposit or ignore, also check for protection
                4. If there is an enemy shipyard: depending on the situation might attack it
        """
        self.weight_convert()
        self.shipyard_status()
        # list of all directions in 5 moves apart
        dirs = list(self.grid.columns)
        # Iterate through different directions
        for direction in dirs:
            # If there was a ship
            if not pd.isna(self.grid[direction].ship_id):
                # If it was my ship
                if self.grid[direction].my_ship == 1:
                    self.avoid_self_colision(self.grid[direction].ship_id)
                else: 
                    # If it was not my ship 
                    self.deal_enemy_ship(self.grid[direction].ship_id)
            
            # If there was a shipyard
            if not pd.isna(self.grid[direction].shipyard_id):
                # If it was my shipyard
                if self.grid[direction].my_shipyard == 1:
                    self.deposit(self.grid[direction].shipyard_id)
                else: 
                    # If it was not my shipyard
                    self.attack_enemy_shipyard(self.grid[direction].shipyard_id)

            # Just to trigger some movement in the main four directions given that they were no ship/shipyards
            if 'N' in direction: self.weights['N'] += self.grid[direction].halite / (self.grid[direction].moves - 0.5)
            if 'W' in direction: self.weights['W'] += self.grid[direction].halite / (self.grid[direction].moves - 0.5)
            if 'E' in direction: self.weights['E'] += self.grid[direction].halite / (self.grid[direction].moves - 0.5)
            if 'S' in direction: self.weights['S'] += self.grid[direction].halite / (self.grid[direction].moves - 0.5)

            # In order to keep the mining as an option as well: the mining option will get add to only if amount of grid in
            # the current cell is higher than other places
            self.weights['mine'] += (self.current_cell.halite - self.grid[direction].halite) / (self.grid[direction].moves - 0.5)


    def weight_convert(self, threshold=2000):
        """ Weights the option for ship convertion. """
        # 1. If they are no shipyards left
        no_yards_left = len(self.player.shipyards) == 0
        # 2. If it is the end of the game and we have more than 500 halite in our cargo
        end_of_game_conversion = (self.step > 395 or self.near_end()) and self.ship.halite >= 500
        # 3. There will be a threshold for the amount of cargo any ship could have
        threshhold_reach = self.ship.halite > threshold
        # 4. On shipyard already
        on_shipyard = self.ship.cell.shipyard != None
        
        if (no_yards_left or end_of_game_conversion or threshhold_reach) and not on_shipyard: 
            self.weights['convert'] = 100
        elif not on_shipyard:
            self.weights['convert'] = round((self.ship.halite - 1000) / (self.step // 50 + 1) , 3)
        else:
            self.weights['convert'] = round(2 * (self.ship.halite - 1000) / (self.step // 50 + 1), 3)
    

    def deposit(self, shipyard_id):
        """ Weights the tendency to deposit and adds to the directions which lead to the given shipyard. """
        # Get the ship's info
        shipyard = self.board.shipyards[shipyard_id]
        moves = (self.Shipyards[shipyard_id].moves - 0.5) # Smoothing
        # Don't get too far from the shipyard
        log(self.Shipyards)
        closest_shipyard_id = self.Shipyards.T['moves'].idxmin()
        closest_shipyard_dist = (self.Shipyards[closest_shipyard_id].moves + 0.7) # Smoothing
        
        # Implement number of ships surrounding the shipyard
        dirX, dirY = self.determine_directions(self.current_position, shipyard.position)
        
        if dirX != None: self.weights[dirX] += 10 * (self.ship_cargo + 3) / closest_shipyard_dist
        if dirY != None: self.weights[dirX] += 10 * (self.ship_cargo + 3) / closest_shipyard_dist
    
    
    def attack_enemy_shipyard(self, shipyard_id):
        """ Weights the tendency to attack the enemy shipyard. """
        # Get the ship's info
        shipyard = self.Shipyards[shipyard_id]
        oppCargo = ship.halite
        moves = (self.Shipyards[shipyard_id].moves - 0.5) # Smoothing
        
        # Implement number of ships surrounding the shipyard
        dirX, dirY = self.determine_directions(self.current_position, shipyard.position)
        
        # Don't get too far from the shipyard
        closest_shipyard_id = self.Shipyards.T['moves'].idxmin()
        closest_shipyard_dist = (self.Shipyards[closest_shipyard_id].moves + 0.99) # Smoothing
        
        if dirX != None: self.weights[dirX] += 10 * oppCargo / closest_shipyard_dist
        if dirY != None: self.weights[dirX] += 10 * oppCargo / closest_shipyard_dist
    
    
    def deal_enemy_ship(self, ship_id):
        """ This function will be evaluate to either attack or get_away from the ship based on
        the simple observation: if my ship had more cargo then I should not attack. """
        # The amount of cargo caried by the opponent's ship
        oppCargo = self.Ships[ship_id].cargo
        
        if self.ship_cargo > oppCargo:
            # If I had more halite then should get away If the ship was close enough then move
            self.get_away(ship_id)
        else:
            # If the ship had more halite then attack, it should take into account
            # how far is it from the nearest shipyard and also how many ships have 
            # surrounded that shipyard
            if not self.Shipyards.empty:
                closest_shipyard_id = self.Shipyards.T['moves'].idxmin()
                closest_shipyard_dist = self.Shipyards[closest_shipyard_id].moves
                self.attack_enemy_shipyard()
    
    
    def attack_enemy_ship(self, ship_id):
        # Get the ship's info
        ship = self.Ships[ship_id]
        oppCargo = ship.halite
        moves = (self.Ships[ship_id].moves - 0.5) # Smoothing
        # Don't get too far from the shipyard
        closest_shipyard_id = self.Shipyards.T['moves'].idxmin()
        closest_shipyard_dist = (self.Shipyards[closest_shipyard_id].moves + 0.99) # Smoothing
        
        # Implement number of ships surrounding the shipyard
        dirX, dirY = self.determine_directions(self.current_position, ship.position)
        
        if dirX != None: self.weights[dirX] += 10 * oppCargo / closest_shipyard_dist
        if dirY != None: self.weights[dirX] += 10 * oppCargo / closest_shipyard_dist
        
    
    def get_away(self, ship_id):
        """ This function is called when our ship has a possible change of getting followed
        by another enemy ship with lower halite.
        There can multiple options that given the parameters and the manual weightings one
        could have a higher value and hence tendency.
        """
        # Get the ship's info
        ship = self.board.ships[ship_id]
        oppCargo = ship.halite
        moves = (self.Ships[ship_id].moves - 0.5) # Smoothing
        
        # Special case: given that the ships is one move away should not mine or convert
        if moves == 1:
            self.weights['mine'] += self.ship_cargo * -10
            self.weights['convert'] += self.ship_cargo * -10
        
        # The ideal thing would be to go to the nearest shipyard
        closest_shipyard_id = self.Shipyards.T['moves'].idxmin()
        dirX, dirY = self.determine_directions(self.current_position, ship.position)
        
        if dirX != None: self.weights[dirX] += 10 * self.ship_cargo / moves
        if dirY != None: self.weights[dirX] += 10 * self.ship_cargo / moves    
    
    
    def shipyard_status(self):
        """ This function gives tendency to go to shipyards within the map. """
        if not self.Shipyards.empty:
            for shipyard_id in list(self.Shipyards.columns):
                self.analyze_shipyard_surroundings(shipyard_id)
        
        
    def analyze_shipyard_surroundings(self, shipyard_id):
        """ Checks to see if a given shipyard needs protection or not? """
        shipyard = self.board.shipyards[shipyard_id]
        
        shipyard_locator = Locator(self.board, shipyard)
        shipyard_grid = shipyard_locator.generate_grid_df()

        for direction in list(shipyard_grid.columns):
            ship_id = shipyard_grid[direction].ship_id
            
            if not pd.isna(ship_id):
                dirX, dirY = self.determine_directions(self.current_position, shipyard.position)
                moves = abs(shipyard_grid[direction]['moves'] - 0.5)
                if shipyard_grid[direction].my_ship == 1:
                    if dirX != None: self.weights[dirX] += -1 * self.ship_cargo / moves
                    if dirY != None: self.weights[dirY] += -1 * self.ship_cargo / moves
                else:
                    if dirX != None: self.weights[dirX] += 3 * self.ship_cargo / moves
                    if dirY != None: self.weights[dirY] += 3 * self.ship_cargo / moves
        
                
    def avoid_self_colision(self, ship_id):
        """ This function is called to avoid the ships collision 
            - The number moves required to reach the ship is also an important factor 
        """
        # This would also encourage for mining
        self.weights['mine'] += 10
        # Getting ship's info
        ship = self.board.ships[ship_id]
        moves = abs(self.Ships[ship_id].moves - 0.5)
        # Getting the direction to the ship
        dirX, dirY = self.determine_directions(self.current_position, ship.position)
        # Should have the highest negative value
        if dirX != None: self.weights[dirX] += -10 * self.ship_cargo / moves
        if dirY != None: self.weights[dirY] += -10 * self.ship_cargo / moves
        

    def determine_directions(self, point1, point2, size=21):
        """ Given two points determine the closest directions to take to get to the point. """
        x1, y1 = point1.x, point1.y
        x2, y2 = point2.x, point2.y

         # For both x and y they are two type of paths to take
        diff_x_1 = abs(x2 - x1) 
        diff_x_2 = abs(size - x2 + x1)
        diff_y_1 = abs(y2 - y1)
        diff_y_2 = abs(size - y2 + y1)
        # Given that x1=x2 or y1=y2 then return None, so we know 
        # no movement was needed in that axis.
        best_x, best_y = None, None

        if diff_x_1 > diff_x_2:
            if x2 - x1 > 0: 
                best_x = "E"
            else:
                best_x = "W"
        else:
            if x2 - x1 > 0: 
                best_x = "W"
            else:
                best_x = "E"

        if diff_y_1 > diff_y_2:
            if y2 - y1 > 0:
                best_y = "N"
            else:
                best_y = "S"
        else:
            if y2 - y1 > 0:
                best_y = "S"
            else:
                best_y = "N"

        return best_x, best_y

    
    def near_end(self):
        """ This function is intended to determine if the game is about to end so the ships with halite can
        convert to shipyard and maximum the halite we will end up with. """
        count = 0
        
        # If the halite was less than 500 and it had no ships
        for opp in self.board.opponents:
            if opp.halite < 500 and len(opp.ships) == 0 and self.player.halite > opp.halite: count += 1
            if opp.halite > 3000 and len(opp.ships) > 1: count -= 1
                
        # If count was more than 2 return True
        return count >= 2
    
class Locator:
    """ This object returns dataframes containing the information about other ships/shipyards on the map and also in the given ship's grid. """
    def __init__(self, board, ship):
        self.board = board
        self.ship = ship
        # Get the grid
        self.grid = grid(ship.cell)
        
    def get_ship_info(self):
        """ Returns the info about ships in all of the board. """
        ships_info = {}
        
        for ship_id, ship in self.board.ships.items():
            base_info = {
                "my_ship": 0, "cargo": 0, 
                "moves": 0, "position": (ship.position.x, ship.position.y)
            }
            
            base_info['cargo'] = ship.cell.halite
            base_info['moves'] = count_moves(self.ship.position, ship.position)
            
            if ship_id in self.board.current_player.ship_ids and ship.id != self.ship.id:
                base_info['my_ship'] = 1
                ships_info[ship_id] = base_info
            elif not(ship_id in self.board.current_player.ship_ids):
                ships_info[ship_id] = base_info
        
        
        return  pd.DataFrame(ships_info)
    
    
    def get_shipyard_info(self):
        """ Returns the info about shipyards in all of the board. """
        shipyards_info = {}
        
        for shipyard_id, shipyard in self.board.shipyards.items():
            base_info = {
                "my_shipyard": 0, "player_halite": 0, 
                "moves": 0, "position": (shipyard.position.x, shipyard.position.y)
            }
            
            base_info['player_halite'] = shipyard.player.halite
            base_info['moves'] = count_moves(self.ship.position, shipyard.position)
            
            if shipyard_id in self.board.current_player.shipyard_ids:
                base_info['my_shipyard'] = 1
                
                shipyards_info[shipyard_id] = base_info
            else:
                shipyards_info[shipyard_id] = base_info
        
        return pd.DataFrame(shipyards_info)
        
        
    def generate_grid_df(self):
        """ Generates a Dataframe describing the information of objects and cells in the 5x5 grid of the ship. """
        all_dirs = {}
        
        for direction, cell in self.grid.items():
            
            base_info = {
                "ship_id": None, "shipyard_id": None, 
                "my_ship": 0, "my_shipyard": 0,
                "halite": 0, "moves": 0
            }
            
            if cell.ship != None: 
                base_info["ship_id"] = cell.ship.id
                if cell.ship.id in self.ship.player.ship_ids:
                    base_info["my_ship"] = 1
    
            if cell.shipyard != None: 
                base_info["shipyard_id"] = cell.shipyard.id
                if cell.shipyard.id in self.ship.player.shipyard_ids:
                    base_info['my_shipyard'] = 1
                    
            base_info['halite'] = cell.halite
            # The number of letters in the direction would indicate the number of moves needed to get there
            base_info['moves'] = len(direction)
            
            all_dirs[direction] = base_info
            
        return pd.DataFrame(all_dirs)
        
####################
# Helper Functions #
####################

def grid(cell):
    """ Returns a dictionary of cells which are in 4 moves distance of the given cell """
    # The directions that are one move away
    north, south, west, east = cell.north, cell.south, cell.west, cell.east
    # The directions that are two moves away
    nn, ss, ww, ee = north.north, south.south, west.west, east.east

    return {
        'N': north, 'S': south, 'W': west, 'E': east, 

        'NW': north.west, 'NE': north.east, 'SW': south.west, 'SE': south.east, 'WW': ww, 'EE': ee, 'NN': nn, 'SS': ss ,
        
        'NEN': nn.east, 'NWN': nn.west, 'SES': ss.east, 'SWS': ss.west, 
        'SEE': ee.south, 'NEE': ee.north, 'SWW': ww.south, 'NWW': ww.north, 
        'SSS': ss.south, 'EEE': ee.east, 'WWW': ww.west, 'NNN': nn.north, 
        
        'NNNN': nn.north.north, 'SSSS': ss.south.south, 'WWWW': ww.west.west, 'EEEE': ee.east.east,
        'SEES': ee.south.south , 'NEEN': ee.north.north, 'NWWN': ww.north.north, 'SWWS': ww.south.south,
        'WWWS': ww.west.south, 'EEES': ee.east.south, 'EEEN': ee.east.north, 'WWWN': ww.west.north,
        'SWSS': ss.south.west, 'SESS': ss.south.east, 'NENN': nn.north.east, 'NWNN': nn.north.west 
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
    opt2 = diff_x_1 + diff_y_2
    opt3 = diff_x_2 + diff_y_1
    opt4 = diff_x_2 + diff_y_2
    
    return min(opt1, opt2, opt3, opt4)


def log(text, step=1):
    if step == 0:
        with open("log.txt", "w") as text_file:
            text = str(text) + '\n'
            text_file.write(text)
    else:
        with open("log.txt", "a") as text_file:
            text = str(text) + '\n'
            text_file.write(text)


# Global values
acts = {
    "N": ShipAction.NORTH, 'S': ShipAction.SOUTH,
    'W': ShipAction.WEST , 'E' : ShipAction.EAST,
    'spawn': ShipyardAction.SPAWN, 'convert': ShipAction.CONVERT,
    'mine': None
}

log('logs:', 0)

def agent(obs, config):
    # Make the board
    board = Board(obs,config)
    #Step of the board
    step = board.observation['step']
    # Current player info
    me = board.current_player # Player Object
    
    new_board = Board(obs,config)
    log(step)
    for ship in me.ships:
        log('ship-id:' + ship.id + ', pos:' + str(ship.position))
        decider = Decision_Ship(new_board, new_board.ships[ship.id])
        ship.next_action = decider.determine()
        
        new_board = board.next()
    
    for shipyard in me.shipyards:
        # If there were no ships on the yard
        if new_board.shipyards[shipyard.id].cell.ship == None and step < 392:
            if len(me.ships) == 0:
                shipyard.next_action = acts['spawn']

            if step % 3 == 1:
                shipyard.next_action = acts['spawn']

            if step > 200 and me.halite > 10000 + len(me.ships) * 1000:
                shipyard.next_action = acts['spawn']
        
        new_board = board.next()
        
    return me.next_actions
