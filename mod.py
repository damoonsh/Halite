from kaggle_environments.envs.halite.helpers import Board, ShipAction, ShipyardAction
import pandas as pd
import numpy as np


class DecisionShip:
    """ 
        Decides ship's next move
        params:
            board: the board that we will base our decisions on
            ship: the ship we are deciding for
            step: the steps into the stimulation
        returns:
            determine: returns the next-action that should be taken
    """
    def __init__(self, board: Board, ship_id, step):
        # The passed variables
        self.board = board
        self.ship = board.ships[ship_id]
        self.step = step
        # Some usefull properties
        self.player = self.board.current_player
        self.ship_cargo = self.ship.halite
        self.current_cell = self.ship.cell
        self.current_position = self.ship.position
        # Check to see if the stimulation is about to end
        self.NEAR_END = self.near_end()
        # All moves ship can take
        self.moves = {"N": ShipAction.NORTH, 'S': ShipAction.SOUTH, 'W': ShipAction.WEST,
                      'E': ShipAction.EAST, 'convert': ShipAction.CONVERT, 'mine': None}
        # The list of moves that should not be taken
        self.eliminated_moves = ['None']
        # Weights of different moves
        self.weights = {"N": 0, "E": 0, "W": 0, "S": 0, "mine": 0, "convert": 0, 'None': 0}
        # The cells around the main one
        self.grid = grid(self.ship.cell)
        # Closest shipyard id and the distance
        self.closest_shipyard_id, self.closest_shipyard_distance = self.closest_shipyard()
        # Default move which is set to mining (None)
        self.next_move = None
        # This variable holds info on the cell being analyzed
        self.current = {}
        # Dictionary indicating the actions that should be taken by other ships to increase the prediction accuracy
        self.other_actions = {}
        # Setting the hyper parameters
        self.set_hyperparameters()

    def set_hyperparameters(self):
        """ Initializes the hyperparameters that will affect the decision process """
        self.MINING = self.mining_hyper()
        self.DEPOSIT = self.deposit_hyper()
        self.DIRECTION_ENCOURAGEMENT = self.direction_encouragement_hyper()
        self.ATTACK_ENEMY_SHIP = self.attack_enemy_ship_hyper()
        self.DISTRIBUTION = self.distribution_hyper()
        self.GET_AWAY = self.get_away_hyper()
        self.CLOSEST_SHIPYARD = self.closest_shipyard_hyper()
        self.CONVERSION = max(60 - len(self.player.shipyards) * 10, 5)

        log('  mining: ' + str(round(self.MINING,2)) + ", conv: " + str(self.CONVERSION) + ", depo: " + str(round(self.DEPOSIT,2)) + ", dir: " + str(round(self.DIRECTION_ENCOURAGEMENT, 2)) + ", att-en-ship: " + str(round(self.ATTACK_ENEMY_SHIP, 2)) + ", distro: " + str(round(self.DISTRIBUTION, 2)) + ', get-away: ' + str(round(self.GET_AWAY, 2)) + ', c_yard: ' + str(round(self.CLOSEST_SHIPYARD, 2)) + ',nearEND: ' + str(self.NEAR_END))
    
    def mining_hyper(self):
        """ Calculates the hyperparameters value for MINING, indirect with ship's cargo """
        return 10 + self.step // 40

    def deposit_hyper(self):
        """ Calculates the hyperparameters value for DEPOSIT, indirect with ship's cargo and step """
        dir_cargo = self.ship_cargo // 300
        dir_step = self.step // 100 + 1
        return 5 + 5 * dir_cargo * dir_step / (self.closest_shipyard_distance + 1) + int(self.NEAR_END) * 5000

    def direction_encouragement_hyper(self):
        """ Calculates the hyperparameters value for DIRECTION_ENCOURAGEMENT, indirect with ship's cargo and direction with step """
        indir_cargo = self.ship_cargo // 250 + 1
        dir_step = 400 - self.step

        return 1e2 * dir_step / indir_cargo

    def get_away_hyper(self):
        """ Calculates the hyperparameters value for DIRECTION_ENCOURAGEMENT, direct with cargo and indirect with number of ships """
        dir_cargo = self.ship_cargo // 50 + 1
        return -10 * dir_cargo

    def attack_enemy_ship_hyper(self):
        """ Calculates the hyperparameters value for ATTACK_ENEMY_SHIP, indirect with cargo and step """
        indir_cargo = (self.ship_cargo // 100) + 1
        indir_step = (self.step // 50) + 1

        return 100 / (indir_cargo * indir_step)

    def distribution_hyper(self):
        """ Calculates the hyperparameters value for DISTRIBUTION, indirect with cargo and step """
        return -20 / ((self.step // 20 + 1) * (self.ship_cargo // 100 + 1))

    def closest_shipyard_hyper(self):
        """ Calculates the hyperparameters value for CLOSEST_SHIPYARD, direct with cargo and step """
        return (self.step // 25 + 1)

    def determine(self):
        """ Returns next action decided for the ship based on the observations that have been made. """
        self.weight_moves()  # Calculate the weights for main four directions
        self.round()  # Round the weights
        self.apply_elimination()  # Apply the eliminations

        # Sort the values
        sorted_weights = {k: v for k, v in sorted(self.weights.items(), key=lambda item: item[1], reverse=True)}

        log('  -> weights: ' + str(sorted_weights))

        # Choose the action with highest value given that it is not eliminated
        for action in sorted_weights.keys():
            return self.moves[action], action

        # If none were chosen, then just return the default move which is mining
        return self.next_move, 'mine'

    def add_accordingly(self, value, title="", loging=False):
        """ Adds a value to directions according to their corresponding weights """
        weightX, weightY, dirX, dirY, movesX, movesY = 0, 0, "", "", 0, 0

        if "N" in self.current['dir']:
            dirY, movesY = 'N', self.current['dir'].upper().count("N")
            weightY = 1 / (len(self.current['dir']) ** 2 * movesY)
        elif "S" in self.current['dir']:
            dirY, movesY = 'S', self.current['dir'].upper().count("S")
            weightY = 1 / (len(self.current['dir']) ** 2 * movesY)

        if "W" in self.current['dir']:
            dirX, movesX = 'W', self.current['dir'].upper().count("W")
            weightX = 1 / (len(self.current['dir']) ** 2 * movesX)
        elif "E" in self.current['dir']:
            dirX, movesX = 'E', self.current['dir'].upper().count("E")
            weightX = 1 / (len(self.current['dir']) ** 2 * movesX)

        if value != 0 and loging:
            log('   ' + title + ', adding ' + str(round(value * weightX, 3)) + ' to ' + dirX + ', moves: ' + str(movesX) + ' and ' + str(round(value * weightY, 2)) + ' to ' + dirY + ", moves: " + str(movesY))

        if weightX != 0: self.weights[dirX] += value * weightX
        if weightY != 0: self.weights[dirY] += value * weightY
        
    def weight_convert(self, base_threshold=600):
        """ Weights the CONVERT option"""
        # Calculating the threshhold
        threshold = base_threshold + 500 * (len(self.player.shipyards) // 4)
        # If they are no shipyards left
        no_shipyards = len(self.player.shipyards) == 0
        # On shipyard already
        on_shipyard = self.ship.cell.shipyard is not None

        if self.player.halite + self.ship.halite >= 500:
            if no_shipyards and not on_shipyard:
                self.weights['convert'] = 1e8 # Very high number that will ensure that 
            elif not on_shipyard:
                # When the ship is not on any shipyard then weight it
                self.weights['convert'] = self.CONVERSION * (self.ship_cargo - threshold) / ((10.99 - self.closest_shipyard_distance) ** 2 * len(self.player.shipyards))
            elif on_shipyard:
                # If the ship was already on a shipyard then eliminate the move
                self.eliminated_moves.append('convert')
        else:
            # If the sum of cargo and player's own halite was less than 500 then eliminate the move
            self.eliminated_moves.append('convert')

    def weight_moves(self):
        """ This is the main function and runs other helper functions within the module to to weight the different moves that could be taken. """
        
        # Weight the CONVERT option
        self.weight_convert()

        # See if any of the shipyards need defending
        self.shipyard_status()

        # Performance issues
        interval = min(220, 12000 // (len(self.player.ships) + 1))

        # Iterate through different directions
        for direction, cell in list(self.grid.items())[:interval]:
            # Set the global values that will be used
            self.current['dir'] = direction
            self.current['cell'] = cell

            # 1. Evaluate the moves based on other objects in the map
            # 1.1 If there was a ship
            if cell.ship is not None:
                if cell.ship.id in self.player.ship_ids:
                    self.distribute_ships(cell.ship_id) # If it was my ship
                else:
                    self.deal_enemy_ship(cell.ship_id)

            # 1.2 If there was a shipyard
            if cell.shipyard is not None:
                if cell.shipyard.id in self.player.shipyard_ids:
                    self.deposit()
                else:
                    self.attack_enemy_shipyard(cell.shipyard_id)

            # 2. Trigger movement in the main four direction solely based on the amount of halite each cell has
            # Estimate how much halite a cell will have then divide it by four, if there was a ship divide by 100
            ship_affect = (1 + int(self.ship is None)) / 200
            main_dir_encourage = self.DIRECTION_ENCOURAGEMENT * (cell.halite * 1.25 ** len(direction)) * ship_affect / 4  
            self.add_accordingly(main_dir_encourage, title='  main4', loging=False)

            # 3. Either encourage mining or discourage it by adding the difference between cells to the mine
            mining_trigger = 10 * (self.current_cell.halite - self.grid[direction].halite * 1.25 ** len(direction)) / (len(direction) ** 2)
            if self.step > 100 and self.step < 104: log('  trigger: ' + str(mining_trigger)) 
            self.weights['mine'] += mining_trigger

        # The correlation of the mining with cell's halite
        if self.step > 100 and self.step < 104: log(" cell_halite: " + str(self.current_cell.halite))
        self.weights['mine'] += self.MINING * self.current_cell.halite ** 2 

    def distribute_ships(self, ship_id):
        """ This function lowers the ships tendency to densely populate an area """
        # Preventing my ships from crashing with each other
        if len(self.current['dir']) == 1: self.eliminated_moves.append(self.current['dir'])
        
        # Encourage distribution
        distribution_encouragement = self.DISTRIBUTION * abs(self.ship_cargo - self.board.ships[ship_id].halite)
        self.add_accordingly(distribution_encouragement, title='Distribution', loging=self.step < 20 or (self.step < 390 and self.step > 385))

    def deal_enemy_ship(self, ship_id):
        """ This function will evaluate to either attack or get_away from an enemy ship based on the 
        simple observation: If my ship had more cargo then I should not attack. """
        if self.ship_cargo > (self.board.ships[ship_id].halite + 0.25 * len(self.current['dir']) * self.current['cell'].halite):
            self.get_away(cargo_diff=abs(self.board.ships[ship_id].halite - self.ship_cargo))
        else:
            self.attack_enemy_ship(abs(self.board.ships[ship_id].halite - self.ship_cargo))

    def get_away(self, cargo_diff=1):
        """ This function is called when my ship needs to get away from a ship which might be following it """
        direction_discouragement = 0

        # 1. Directly discouraging the movement
        if len(self.current['dir']) == 1:
            self.eliminated_moves.append(self.current['dir'])
            if 'mine' in self.weights.keys(): self.eliminated_moves.append('mine')    
        else:
            direction_discouragement = self.GET_AWAY * (self.ship.halite  + 0.1)
        
        self.add_accordingly(direction_discouragement, title='Get-Away', loging=True)

        # 2. Encourage going to the closest shipyard
        closest_shipyard_encouragement = self.CLOSEST_SHIPYARD * cargo_diff ** 1.5 / len(self.current['dir']) ** 2
        
        self.go_to_closest_shipyard(closest_shipyard_encouragement)

    def go_to_closest_shipyard(self, value):
        """ Encourage movement towards the nearest shipyard """
        
        if self.closest_shipyard_id != 0.99: # Given that there is a closest shipyard
            (x, y) = self.board.shipyards[self.closest_shipyard_id].position
            
            if x > self.current_cell.position.x:
                self.weights['E'] += value 
                log('   closest_yard: ' + str(value) + " at E")
            elif x < self.current_cell.position.x:
                self.weights['W'] += value
                log('   closest_yard: ' + str(value) + " at W")

            if y > self.current_cell.position.y:
                self.weights['N'] += value
                log('   closest_yard: ' + str(value) + " at N")
            elif y < self.current_cell.position.y:
                self.weights['S'] += value 
                log('   closest_yard: ' + str(value) + " at S")

    def attack_enemy_ship(self, diff):
        """ This function encourages attacking the enemy ship """
        attack_encouragement = self.ATTACK_ENEMY_SHIP * (diff + 1) / (self.closest_shipyard_distance + 0.1)
        self.add_accordingly(attack_encouragement, title='Attacking-Enemy-Ship', loging=True)
        
        # If the enemy ship was one move away then update it so the other ships could see it 
        # if len(self.current['dir']) == 1:
            # pass
    
    def basic_self_move(self, cell):
        """ Decides a basic move for one of my ships that is about to get hit by an enemy ship. """
        pass

    def basic_enemy_move(self, main_cell):
        """ Decides a basic move for an enemy ship which is about to get hit by one of my own ships. """
        pass

    def deposit(self):
        """ Weights the tendency to deposit and adds to the directions which lead to the given shipyard """
        deposit_tendency = self.DEPOSIT * self.ship_cargo
        self.add_accordingly(deposit_tendency, title='Deposit', loging=self.step > 30 and self.step < 50)

    def attack_enemy_shipyard(self, shipyard_id):
        """ Weights the tendency to attack the enemy shipyard. """
        dist_to_shipyard = measure_distance(self.board.shipyards[shipyard_id].position, self.current['cell'].position)
        if len(self.player.ships) >= 2 and self.player.halite > 700 and self.ship_cargo < 30 and dist_to_shipyard < 5:
            destory_shipyard = 1e6 / len(self.current['dir']) ** 2
            self.add_accordingly(destory_shipyard, title='  Destroy_en_shipyard', loging=True)
        elif len(self.current['dir']) == 1 and self.ship_cargo > 100:
            self.eliminated_moves.append(self.current['dir'])

    def shipyard_status(self):
        """ Measures tendency for the shipyards within the map """
        if len(self.player.shipyards) != 0:
            for shipyard in self.player.shipyards:
                self.analyze_shipyard_surroundings(shipyard.id)

    def analyze_shipyard_surroundings(self, shipyard_id):
        """ Analyzes the tendency to go toward a specific shipyard """
        shipyard, value = self.board.shipyards[shipyard_id], 0
        shipyard_grid = grid(shipyard.cell) # Limit it to four moves away

        for cell in shipyard_grid.values():
            # If there is a ship on that cell
            if cell.ship is not None:
                if cell.ship.id in self.player.ship_ids:
                    value += -1e4 / (self.ship_cargo + 0.99)
                else:
                    value += 1e4 /(cell.ship.halite + 0.99) 

        # Don't discourage any move toward a shipyards
        if value > 0:
            currentDir = ""

            if shipyard.position.x > self.current_position.x:
                currentDir += "E" * abs(shipyard.position.x - self.current_position.x)
            elif shipyard.position.x < self.current_position.x:
                currentDir += "W" * abs(shipyard.position.x - self.current_position.x)

            if shipyard.position.y > self.current_position.y:
                currentDir += "S" * abs(shipyard.position.y - self.current_position.y)
            elif shipyard.position.y < self.current_position.y:
                currentDir += "N" * abs(shipyard.position.y - self.current_position.y)

            if currentDir != "":
                self.current['dir'] = currentDir
                self.add_accordingly(value, title='  Yard-sur', loging=True)

    def closest_shipyard(self):
        """ Returns the closest shipyard's id """
        closest_id, diff = 0.99, 0.99
        for shipyard in self.player.shipyards:
            distance = measure_distance(self.current_cell.position, shipyard.cell.position)
            if diff > distance or diff == 0.99:
                closest_id, diff = shipyard.id, distance
        return closest_id, diff
        
    def near_end(self):
        """ Determines if the game is about to end so the ships with halite can convert to shipyard and maximum the halite we will end up with """
        count = 0

        # If the halite was less than 500 and it had no ships
        for opp in self.board.opponents:
            if opp.halite < 500 and len(opp.ships) == 0 and self.player.halite > opp.halite: count += 1
            if opp.halite > 2000 and len(opp.ships) > 1: count -= 1

        # If count was more than 2 return True
        return (count >= 2 or self.step > 385)

    def apply_elimination(self):
        """ Eliminates the moves to be eliminated. """
        for move in self.eliminated_moves:
            if move in self.weights.keys():
                del self.weights[move]

    def round(self):
        """ This functions rounds the weights so they can be easily printed """
        self.weights['mine'] = round(self.weights['mine'], 1)
        self.weights['N'] = round(self.weights['N'], 1)
        self.weights['S'] = round(self.weights['S'], 1)
        self.weights['E'] = round(self.weights['E'], 1)
        self.weights['W'] = round(self.weights['W'], 1)
        self.weights['convert'] = round(self.weights['convert'], 1)


class ShipyardDecisions:
    def __init__(self, board: Board, player, step):
        """
            Decides the Shipyard's next action based on the given parameters
            board: The board that we will be observing
            step: step of the stimulation
        """
        self.board = board
        self.player = player
        self.player_halite = player.halite
        self.step = step
        self.Shipyards = player.shipyards
        self.shipyard_tendencies = {}

    def determine(self):
        """ Determines which shipyards should SPAWN, returns a dictionary of id: 'SPAWN' """
        self.weight_shipyard_tendencies()
        sorted_weights = {k: v for k, v in
                          sorted(self.shipyard_tendencies.items(), key=lambda item: item[1], reverse=True)}
        log(' Shipyard weights: ' + str(sorted_weights))
        shipyard_ids = []
        for shipyard_id, tendency in sorted_weights.items():
            if tendency > 5 and self.player_halite >= 500:
                shipyard_ids.append(shipyard_id)

        # log('Shipyards: ' + str(shipyard_ids))
        return shipyard_ids

    def weight_shipyard_tendencies(self):
        """ Iterates through the shipyards and weights their tendencies. """
        for shipyard in self.Shipyards:
            # Weighting will take place only when there are no ships on the cell
            if shipyard.cell.ship is None:
                weight = self.weight(grid(shipyard.cell))

                self.shipyard_tendencies[shipyard.id] = weight

    def weight(self, grid):
        """
            Weights shipyard's tendency to spawn solely based on the objects around it
            The weighting system is rather simple:
                - If there was an enemy ship add to the weight
                - If there was one of my own ships, then subtract from the weight
            Take the distance of the ship into account
        """
        # If I had no ships then SPAWN
        if len(self.board.current_player.ships) == 0: return 100
        # Get the averages
        if self.step < 50 and self.player_halite >= 500: return 100

        value = 0
        # Iterating through the grid
        for direction, cell in grid.items():
            if cell.ship is not None:
                if cell.ship.id in self.player.ship_ids:
                    value -= 11 / len(direction) ** 2
                else:
                    value += 10 / len(direction) ** 2
                    # If there was an enemy ship one move away from my shipyard then spawn
                    if len(direction) == 1 and self.player_halite > 500 and cell.ship.halite < 100: 
                        value += 1e3 

            if cell.shipyard is not None:
                if cell.shipyard.id in self.player.shipyard_ids:
                    value += 10 / len(direction) ** 2

        return round(value, 2)

def measure_distance(org, dest):
    """ Measures the distance between two points """
    x_1 = abs(org.x - dest.x)
    x_2 = abs(21 - org.x + dest.x)
    y_1 = abs(org.y - dest.y)
    y_2 = abs(21 - org.y + dest.y)

    return min((x_1 + y_1), (x_1 + y_2), (x_2 + y_1), (x_2 + y_2))

# def determine_direction():
#      if shipyard.position.x > self.current_position.x:
#                 currentDir += "E" * abs(shipyard.position.x - self.current_position.x)
#             elif shipyard.position.x < self.current_position.x:
#                 currentDir += "W" * abs(shipyard.position.x - self.current_position.x)

#             if shipyard.position.y > self.current_position.y:
#                 currentDir += "S" * abs(shipyard.position.y - self.current_position.y)
#             elif shipyard.position.y < self.current_position.y:
#                 currentDir += "N" * abs(shipyard.position.y - self.current_position.y)

def grid(cell):
    """ Returns a dictionary of cells which are in 10 moves distance of the given cell """
    # The directions that are one move away
    north, south, west, east = cell.north, cell.south, cell.west, cell.east
    # The directions that are two moves away
    n2, s2, w2, e2 = north.north, south.south, west.west, east.east
    n3, s3, w3, e3 = n2.north, s2.south, w2.west, e2.east
    n4, s4, w4, e4 = n3.north, s3.south, w3.west, e3.east
    n5, s5, w5, e5 = n4.north, s4.south, w4.west, e4.east
    n6, s6, w6, e6 = n5.north, s5.south, w5.west, e5.east
    n7, s7, w7, e7 = n6.north, s6.south, w6.west, e6.east
    n8, s8, w8, e8 = n7.north, s7.south, w7.west, e7.east
    n9, s9, w9, e9 = n8.north, s8.south, w8.west, e8.east
    n10, s10, w10, e10 = n9.north, s9.south, w9.west, e9.east

    return {
        # 1 move away
        'N': north, 'S': south, 'W': west, 'E': east,
        # 2 moves away
        'NW': north.west, 'NE': north.east, 'SW': south.west, 'SE': south.east, 'WW': w2, 'EE': e2, 'NN': n2, 'SS': s2,
        # 3 moves away
        'SSS': s3, 'EEE': e3, 'WWW': w3, 'NNN': n3,
        'ENN': n2.east, 'WNN': n2.west, 'ESS': s2.east, 'WSS': s2.west,
        'SEE': e2.south, 'NEE': e2.north, 'SWW': w2.south, 'NWW': w2.north,
        # 4 moves away
        'NNNN': n4, 'SSSS': s4, 'WWWW': w4, 'EEEE': e4,
        'EESS': e2.south.south, 'EENN': e2.north.north, 'WWNN': w2.north.north, 'WWSS': w2.south.south,
        'WWWS': w3.south, 'EEES': e3.south, 'EEEN': e3.north, 'WWWN': w3.north,
        'SSSW': s3.west, 'SSSE': s3.east, 'NNNE': n3.east, 'NNNW': n3.west,
        # 5 moves away
        'SSSSS': s5, 'NNNNN': n5, 'WWWWW': w5, 'EEEEE': e5,
        'WWWWN': w4.north, 'WWWWS': w4.south, 'EEEEN': e4.north, 'EEEES': e4.south,
        'SSSSE': s4.east, 'SSSSW': s4.west, 'NNNNW': n4.west, 'NNNNE': n4.east,
        'EESSS': s3.east.east, 'WWSSS': s3.west.west, 'EENNN': n3.east.east, 'WWNNN': n3.west.west,
        'EEESS': e3.south.south, 'EEENN': e3.north.north, 'WWWSS': w3.south.south, 'WWWNN': w3.north.north,
        # 6 moves away
        'SSSSSS': s6, 'NNNNNN': n6, 'WWWWWW': w6, 'EEEEEE': e6,
        'WWWWWN': w5.north, 'WWWWWS': w5.south, 'EEEEEN': e5.north, 'EEEEES': e5.south,
        'SSSSSE': s5.east, 'SSSSSW': s5.west, 'NNNNNW': n5.west, 'NNNNNE': n5.east,
        'WWWWNN': w4.north.north, 'WWWWSS': w4.south.south, 'EEEENN': e4.north.north, 'EEEESS': e4.south.south,
        'NNNNEE': n4.east.east, 'NNNNWW': n4.west.west, 'SSSSWW': s4.west.west, 'SSSSEE': s4.east.east,
        'EEENNN': e3.north.north.north, 'EEESSS': e3.south.south.south, 'WWWNNN': w3.north.north.north, 'WWWSSS': w3.south.south.south,
        # 7 moves away
        'SSSSSSS': s7, 'NNNNNNN': n7, 'WWWWWWW': w7, 'EEEEEEE': e7,
        'WWWWWWN': w6.north, 'WWWWWWS': w6.south, 'EEEEEEN': e6.north, 'EEEEEES': e6.south,
        'SSSSSSE': s6.east, 'SSSSSSW': s6.west, 'NNNNNNW': n6.west, 'NNNNNNE': n6.east,
        'WWWWWNN': w5.north.north, 'WWWWWSS': w5.south.south, 'EEEEENN': e5.north.north, 'EEEEESS': e5.south.south,
        'NNNNNWW': n5.west.west, 'NNNNNEE': n5.east.east, 'SSSSSWW': s5.west.west, 'SSSSSEE': s5.east.east,
        'EEEENNN': e4.north.north.north, 'EEEESSS': e4.south.south.south, 'WWWWNNN': w4.north.north.north, 'WWWWSSS': w4.south.south.south,
        'NNNNEEE': n4.east.east.east, 'NNNNWWW': n4.west.west.west, 'SSSSWWW': s4.west.west.west, 'SSSSEEE': s4.east.east.east,
        # 8 moves away
        'SSSSSSSS': s8, 'NNNNNNNN': n8, 'WWWWWWWW': w8, 'EEEEEEEE': e8,
        'WWWWWWWN': w7.north, 'WWWWWWWS': w7.south, 'EEEEEEEN': e7.north, 'EEEEEEES': e7.south,
        'SSSSSSSE': s7.east, 'SSSSSSSW': s7.west, 'NNNNNNNW': n7.west, 'NNNNNNNE': n7.east,
        'WWWWWWNN': w6.north.north, 'WWWWWWWSS': w6.south.south, 'EEEEEEENN': e6.north.north, 'EEEEEEESS': e6.south.south,
        'NNNNNNWW': n6.west.west, 'NNNNNNEE': n6.east.east, 'SSSSSSWW': s6.west.west, 'SSSSSSEE': s6.west.west,
        'NNNNNWWW': n5.west.west.west, 'NNNNNEEE': n5.east.east.east, 'SSSSSWWW': s5.west.west.west, 'SSSSSEEE':  s5.east.east.east,
        'EEEEENNN': e5.north.north.north, 'EEEEESSS': e5.south.south.south, 'WWWWWNNN': w5.north.north.north, 'WWWWWSSS': w5.south.south.south,
        'EEEENNNN': e4.north.north.north.north, 'WWWWNNNN': w4.north.north.north.north, 'EEEESSSS': e4.south.south.south.south, 'WWWWSSSS': w4.south.south.south.south,
        # 9 moves away
        'SSSSSSSSS': s9, 'NNNNNNNNN': n9, 'WWWWWWWWW': w9, 'EEEEEEEEE': e9,
        'WWWWWWWWN': w8.north, 'WWWWWWWWS': w8.south, 'EEEEEEEEN': e8.north, 'EEEEEEEES': e8.south,
        'SSSSSSSSE': s8.east, 'SSSSSSSSW': s8.west, 'NNNNNNNNW': n8.west, 'NNNNNNNNE': n8.east,
        'NNNNNNNEE': n7.east.east, 'NNNNNNNWW': n7.west.west, 'SSSSSSSEE': s7.east.east, 'SSSSSSSWW': s7.west.west,
        'EEEEEEENN': e7.north.north, 'EEEEEEESS': e7.south.south, 'WWWWWWWNN': w7.north.north, 'WWWWWWWSS': w7.south.south,
        'NNNNNNWWW': n6.west.west.west, 'NNNNNNEEE': n6.east.east.east, 'SSSSSSWWW': s6.west.west.west, 'SSSSSSEEE': s6.east.east.east,
        'EEEEEENNN': e6.north.north.north, 'EEEEEESSS': e6.south.south.south, 'WWWWWWNNN': w6.north.north.north, 'WWWWWWSSS': w6.south.south.south,
        'NNNNNWWWW': n5.west.west.west.west, 'NNNNNEEEE': n5.east.east.east.east, 'SSSSSWWWW': s5.west.west.west.west, 'SSSSSEEEE':  s5.east.east.east.east,
        'EEEEENNNN': e5.north.north.north.north, 'EEEEESSSS': e5.south.south.south.south, 'WWWWWNNNN': w5.north.north.north.north, 'WWWWWSSSS': w5.south.south.south.south,
        # 10 moves away
        'SSSSSSSSSS': s10, 'NNNNNNNNNN': n10, 'WWWWWWWWWW': w10, 'EEEEEEEEEE': e10,
        'WWWWWWWWWN': w9.north, 'WWWWWWWWWS': w9.south, 'EEEEEEEEEN': e9.north, 'EEEEEEEEES': e9.south,
        'SSSSSSSSSE': s9.east, 'SSSSSSSSSW': s9.west, 'NNNNNNNNNW': n9.west, 'NNNNNNNNNE': n9.east,
        'NNNNNNNNEE': n8.east.east, 'SSSSSSSSEE': s8.east.east, 'NNNNNNNNWW': n8.west.west, 'SSSSSSSSWW': s8.west.west,
        'EEEEEEEENN': e8.north.north, 'EEEEEEEESS': e8.south.south, 'WWWWWWWWNN': w8.north.north, 'WWWWWWWWNN': w8.north.north,
        'WWWWWWWNNN': w7.north.north.north, 'WWWWWWWSSS': w7.south.south.south, 'EEEEEEESSS': e7.south.south.south, 'EEEEEEEWWW': e7.north.north.north,
        'NNNNNNNWWW': n7.west.west.west, 'NNNNNNNEEE': n7.east.east.east, 'SSSSSSSWWW': s7.west.west.west, 'SSSSSSSEEE': s7.east.east.east,
        'WWWWWWNNNN': w6.north.north.north.north, 'WWWWWWSSSS': w6.south.south.south.south, 'EEEEEENNNN': e6.north.north.north.north, 'EEEEEESSSS': e6.south.south.south.south,
        'NNNNNNWWWW': n6.west.west.west.west, 'NNNNNNEEEE': n6.east.east.east.east, 'SSSSSSWWWW': s6.west.west.west.west, 'SSSSSSEEEE': s6.east.east.east.east,
        'NNNNNWWWWW': n5.west.west.west.west.west, 'NNNNNEEEEE': n5.east.east.east.east.east, 'SSSSSWWWWW': s5.west.west.west.west.west, 'SSSSSEEEEE': s5.east.east.east.east.east
    }


def log(text, step=1):
    if step == 0:
        with open("log.txt", "w") as text_file:
            text = str(text) + '\n'
            text_file.write(text)
    else:
        with open("log.txt", "a") as text_file:
            text = str(text) + '\n'
            text_file.write(text)


log('logs:', 0)

movement_dictionary = {"S": "SOUTH", 'N': 'NORTH', 'W': 'WEST', 'E': 'EAST', 'convert': 'CONVERT'}

import operator

def agent(obs, config):
    # Another for updates
    board = Board(obs, config)

    # Step of the board
    step = board.observation['step']

    ships = [ship.id for ship in sorted(board.current_player.ships, key=operator.attrgetter("halite"), reverse=True)]
    actions = {}

    # It would be absurd to log when I am out of the game
    log(str(step + 1) + '|-----------------------------------------------------------------------')
    log(' Halite: ' + board.current_player.halite + ', n_ships:' + len(board.current_player.ships) + ',n_yards: ' + len(board.current_player.shipyards))
    for ship_id in ships:
        if ship_id in board.current_player.ship_ids:
            log(' Pos:' + str(board.ships[ship_id].position) + ', cargo: ' + str(board.ships[ship_id].halite) + ', player halite: ' + str(board.current_player.halite))
                
            next_action, action_type = DecisionShip(board, ship_id, step).determine()
                
            if action_type != 'mine':
                actions[ship_id] = movement_dictionary[action_type]
                board.ships[ship_id].next_action = next_action
                board = board.next()

    shipyard_ids = ShipyardDecisions(board, board.current_player, step).determine()

    for shipyard_id in board.current_player.shipyard_ids:
        if shipyard_id in shipyard_ids:
            actions[shipyard_id] = 'SPAWN'
            board.shipyards[shipyard_id].next_action = ShipyardAction.SPAWN
            
            board = board.next()
        
    return actions
