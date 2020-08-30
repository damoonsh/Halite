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
        self.board = board
        self.ship = board.ships[ship_id]
        self.step = step

        # Some usefull properties
        self.player = self.board.current_player
        self.ship_cargo = self.ship.halite
        self.current_cell = self.ship.cell
        self.current_position = self.ship.position
        
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
        # Setting the hyper parameters
        self.set_hyperparameters()

    def set_hyperparameters(self):
        """ Initializes the hyperparameters that will affect the decision process """
        self.MINING = 300 + (self.step // 80) * 75
        self.DEPOSIT = 800 + self.step // 300 * 300 + self.step // 380 * 800
        self.ATTACK_ENEMY_SHIP = 400
        self.ATTACK_ENEMY_SHIPYARD = 200
        self.DIRECTION_ENCOURAGEMENT = 80 - (self.step // 50) * 10 - (self.step // 300) * 20 
        self.DISTRIBUTION = -10
        self.GET_AWAY = -50
        self.CLOSEST_SHIPYARD = 500
        self.CONVERSION = 50
        self.CARGO_THRESHHOLD = 600

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
            dirY, movesY = 'N', self.current['dir'].count("N")
            weightY = 1 / (len(self.current['dir']) ** 2 * movesY)
        elif "S" in self.current['dir']:
            dirY, movesY = 'S', self.current['dir'].count("S")
            weightY = 1 / (len(self.current['dir']) ** 2 * movesY)

        if "W" in self.current['dir']:
            dirX, movesX = 'W', self.current['dir'].count("W")
            weightX = 1 / (len(self.current['dir']) ** 2 * movesX)
        elif "E" in self.current['dir']:
            dirX, movesX = 'E', self.current['dir'].count("E")
            weightX = 1 / (len(self.current['dir']) ** 2 * movesX)

        if value != 0 and loging:
            log('   ' + title + ', adding ' + str(round(value * weightX, 3)) + ' to ' + dirX + ', moves: ' + str(movesX) + ' and ' + str(round(value * weightY, 2)) + ' to ' + dirY + ", moves: " + str(movesY))

        if weightX != 0: self.weights[dirX] += value * weightX
        if weightY != 0: self.weights[dirY] += value * weightY
        
    def weight_convert(self, base_threshold=600):
        """ Weights the option for ship conversion """
        # Calculating the threshhold
        threshold = base_threshold + 1000 * (len(self.player.shipyards) // 3)
        # 1. If they are no shipyards left
        no_shipyards = len(self.player.shipyards) == 0
        # 2. There will be a threshold for the amount of cargo any ship could have
        threshhold_reach = self.ship.halite > threshold
        # 3. On shipyard already
        on_shipyard = self.ship.cell.shipyard is not None

        # self.Shipyards[self.closest_shipyard_id]['moves'] < 12

        if self.player.halite + self.ship.halite >= 500:
            if no_shipyards and not on_shipyard:
                self.weights['convert'] = 1e6
            elif not on_shipyard:
                self.weights['convert'] = (self.ship_cargo - threshold) * 60
            elif on_shipyard:
                self.eliminated_moves.append('convert')
        else:
            self.eliminated_moves.append('convert')

    def weight_moves(self):
        """ This is the main function and runs other helper functions within the module to to weight the different moves that could be taken. """
        
        # Weight the CONVERT option
        self.weight_convert()

        # See if any of the shipyards need defending
        self.shipyard_status()

        # Iterate through different directions
        for direction, cell in self.grid.items():
            # Set the global direction to the one at hand
            self.current['dir'] = direction
            self.current['cell'] = cell

            # 1. Evaluate the moves based on other objects present in the map
            # 1.1 If there was a ship
            if cell.ship is not None:
                # If it was my ship
                if cell.ship.id in self.player.ship_ids:
                    self.distribute_ships(cell.ship_id)
                else:
                    self.deal_enemy_ship(cell.ship_id)

            # 1.2 If there was a shipyard
            if cell.shipyard is not None:
                if cell.shipyard.id in self.player.shipyard_ids:
                    self.deposit()
                else:
                    self.attack_enemy_shipyard(cell.shipyard_id)

            # 2. Trigger movement in the main four direction solely based on the amount of halite each cell has
            main_dir_encourage = self.DIRECTION_ENCOURAGEMENT * self.grid[direction].halite + 10
            self.add_accordingly(main_dir_encourage, title='  main4: ', loging=False)

            # 3. Either encourage mining or discourage it by adding the difference between cells to the mine
            mining_trigger = (self.current_cell.halite - self.grid[direction].halite) / len(direction)

            self.weights['mine'] += mining_trigger

        # The correlation of the mining with cell's halite
        self.weights['mine'] += self.current_cell.halite * self.MINING
        # log('  Mining-enc: ' + str(round(self.current_cell.halite ** 2, 2)))

    def distribute_ships(self, ship_id):
        """ This function lowers the ships tendency to densely populate an area """
        if len(self.current['dir']) == 1: self.eliminated_moves.append(self.current['dir'])

        distribution_encouragement = -10 * abs(self.ship_cargo - self.board.ships[ship_id].halite)

        self.add_accordingly(distribution_encouragement, title='Distribution', loging=False)

    def deal_enemy_ship(self, ship_id):
        """ This function will evaluate to either attack or get_away from an enemy ship based on the 
        simple observation: If my ship had more cargo then I should not attack. """
        # If the ship's cargo was more than enemy's cargo and it was not equal to zero then get away otherwise attack
        if self.ship_cargo > (self.board.ships[ship_id].halite + 0.4 * self.current['cell'].halite):
            self.get_away(cargo_diff=abs(self.board.ships[ship_id].halite - self.ship_cargo))
        else:
            self.attack_enemy_ship(abs(self.board.ships[ship_id].halite - self.ship_cargo))

    def attack_enemy_ship(self, diff):
        """ This function encourages attacking the enemy ship """
        attack_encouragement = self.ATTACK_ENEMY_SHIP * (diff + 1) / (self.closest_shipyard_distance + 0.1)
        self.add_accordingly(attack_encouragement, title='Attacking-Enemy-Ship', loging=False)
        
        # If the enemy ship was one move away then update it so the other ships could see it 
        # if len(self.current['dir']) == 1:
            # pass

    # def enemy_ship_runaway(self, enemy_ship_id):
    #     cell = self.board.ships[enemy_ship_id].cell
    #     base_value, preferred_dir = -1e4, ""

    #     for Dir, adj_cell in {"N": cell.north, "S": cell.south, "E": cell.east, "W": cell.west}.items():
    #         weight_adj = cell.north, cell.south, cell.east, cell.west
            
    #         if weight_adj > base_value or preferred_dir == "": base_value, preferred_dir = weight_adj, Dir

    # @staticmethod
    # def weight_enemy_tendency(cell, main_cell):
        
    #     value = 0
    #     if cell.ship is not None:
    #         if cell.ship.id in main_cell.ship.player.ship_ids:
    #             value -= 1e3
    #         else:
    #             value += 10 * (main_cell.ship.halite - cell.ship.halite + 3)

    #     if cell.shipyard is not None:
    #         if cell.shipyard_id in main_cell.ship.player.shipyard_ids:
    #             value += main_cell.ship.halite
    #         else:
    #             value += 10

    #     value += cell.halite * 5
    #     return value

    def get_away(self, cargo_diff=0):
        """ This function is called when my ship needs to get away from a ship which might be following it """
        # 1. Directly discouraging the movement
        if len(self.current['dir']) == 1:
            self.eliminated_moves.append(self.current['dir'])
            self.eliminated_moves.append('mine')
            direction_discouragement = 0
        elif len(self.current['dir']) == 2:
            # When the enemy ship is two moves away, there should be a strong discouragement
            direction_discouragement = 3 * self.GET_AWAY * (self.ship.halite  + 10)
        else:
            direction_discouragement = self.GET_AWAY * cargo_diff
        
        self.add_accordingly(direction_discouragement, title='Get-Away', loging=False)

        # 2. Encouraging going to the closest shipyard
        closest_shipyard_encouragement = cargo_diff
        self.go_to_closest_shipyard(closest_shipyard_encouragement)

    def deposit(self):
        """ Weights the tendency to deposit and adds to the directions which lead to the given shipyard """
        if self.near_end():
            deposit_tendency = 16 * self.DEPOSIT * self.ship_cargo
        else:
            deposit_tendency = self.DEPOSIT * self.ship_cargo 
        self.add_accordingly(deposit_tendency, title='Deposit', loging=False)

    def attack_enemy_shipyard(self, shipyard_id):
        """ Weights the tendency to attack the enemy shipyard. """
        if len(self.player.ships) >= 2 and self.player.halite > 700 and self.ship_cargo < 30 and self.closest_shipyard_distance < 5:
            destory_shipyard = 1e4 / len(self.current['dir'])
            self.add_accordingly(destory_shipyard, title='Destroy_en_shipyard', loging=False)
        elif len(self.current['dir']) == 1 and self.ship_cargo > 100:
            self.eliminated_moves.append(self.current['dir'])

    def go_to_closest_shipyard(self, value):
        """ Encourage movement towards the nearest shipyard """
        if self.closest_shipyard_id != 0.99: # Given that there is a closest shipyard
            (x, y) = self.board.shipyards[self.closest_shipyard_id].position

            if x > self.current_cell.position.x:
                self.weights['E'] += value
            elif x < self.current_cell.position.x:
                self.weights['W'] += value

            if y > self.current_cell.position.y:
                self.weights['N'] += value
            elif y < self.current_cell.position.y:
                self.weights['S'] += value

    def shipyard_status(self):
        """ Measures tendency for the shipyards within the map """
        if len(self.player.shipyards) != 0:
            for shipyard in self.player.shipyards:
                self.analyze_shipyard_surroundings(shipyard.id)

    def analyze_shipyard_surroundings(self, shipyard_id):
        """ Analyzes the tendency to go toward a specific shipyard """
        shipyard, value = self.board.shipyards[shipyard_id], 0
        shipyard_grid = grid(shipyard.cell)

        for direction, cell in shipyard_grid.items():
            ship_id = shipyard_grid[direction].ship_id
            # If there is a ship on that cell
            if cell.ship is not None:
                if cell.ship.id in self.player.ship_ids:
                    value += -1e3
                else:
                    value += 1e3 /(cell.ship.halite + 0.5) 

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
                log("current direction: " + currentDir)
                self.current['dir'] = currentDir
                self.add_accordingly(value, title='  Yard-sur', loging=True)

    def closest_shipyard(self):
        """ Returns the closest shipyard's id """
        closest_id, diff = 0.99, 0.99
        for shipyard in self.player.shipyards:
            distance = self.measure_distance(shipyard.cell)
            if diff > distance or diff == 0.99:
                closest_id, diff = shipyard.id, distance
        return closest_id, diff
    
    def measure_distance(self, dest):
        """ Measures the distance between two points """
        x_1 = abs(self.current_cell.position.x - dest.position.x)
        x_2 = abs(21 - self.current_cell.position.x + dest.position.x)
        y_1 = abs(self.current_cell.position.y - dest.position.y)
        y_2 = abs(21 - self.current_cell.position.y + dest.position.y)

        return min((x_1 + y_1), (x_1 + y_2), (x_2 + y_1), (x_2 + y_2))
    
    def near_end(self):
        """ Determines if the game is about to end so the ships with halite can convert to shipyard and maximum the halite we will end up with """
        count = 0
        # If the halite was less than 500 and it had no ships
        for opp in self.board.opponents:
            if opp.halite < 500 and len(opp.ships) == 0 and self.player.halite > opp.halite: count += 1
            if opp.halite > 2000 and len(opp.ships) > 1: count -= 1
        # If count was more than 2 return True
        return count >= 2

    def apply_elimination(self):
        """ Eliminates the moves to be eliminated. """
        for move in self.eliminated_moves:
            if move in self.weights.keys():
                del self.weights[move]

    def round(self):
        """ This functions rounds the  weights so they can be easily printed """
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
            if tendency > 0 and self.player_halite >= 500:
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
        if len(self.board.current_player.ships) == 0:
            return 10
        if self.step < 70 and self.player_halite >= 500:
            return 10

        value = 0
        # Iterating through the grid
        for direction, cell in grid.items():
            if cell.ship is not None:
                if cell.ship.id in self.player.ship_ids:
                    value -= 120 / len(direction)
                else:
                    value += 100 / len(direction)
                    # If there was an enemy ship one move away from my shipyard then spawn
                    if len(direction) == 1 and self.player_halite > 500: 
                        value += 1e3 

            if cell.shipyard is not None:
                if cell.shipyard.id in self.player.shipyard_ids:
                    value += 200 / len(direction)

        return round(value, 2)

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
        'SSSSSSSS': s9, 'NNNNNNNN': n9, 'WWWWWWWW': w9, 'EEEEEEEE': e9,
        'WWWWWWWWN': w8.north, 'WWWWWWWWS': w8.south, 'EEEEEEEEN': e8.north, 'EEEEEEEES': e8.south,
        'SSSSSSSSE': s8.east, 'SSSSSSSSW': s8.west, 'NNNNNNNNW': n8.west, 'NNNNNNNNE': n8.east,
        'NNNNNNNEE': n7.east.east, 'NNNNNNNWW': n7.west.west, 'SSSSSSSEE': s7.east.east, 'SSSSSSSWW': s7.west.west,
        'EEEEEEENN': e7.north.north, 'EEEEEEESS': e7.south.south, 'WWWWWWWNN': w7.north.north, 'WWWWWWWSS': w7.south.south,
        'NNNNNNWWW': n6.west.west.west, 'NNNNNNEEE': n6.east.east.east, 'SSSSSSWWW': s6.west.west.west, 'SSSSSSEEE': s6.east.east.east,
        'EEEEEENNN': e6.north.north.north, 'EEEEEESSS': e6.south.south.south, 'WWWWWWNNN': w6.north.north.north, 'WWWWWWSSS': w6.south.south.south,
        'NNNNNWWWW': n5.west.west.west.west, 'NNNNNEEEE': n5.east.east.east.east, 'SSSSSWWWW': s5.west.west.west.west, 'SSSSSEEEE':  s5.east.east.east.east,
        'EEEEENNNN': e5.north.north.north.north, 'EEEEESSSS': e5.south.south.south.south, 'WWWWWNNNN': w5.north.north.north.north, 'WWWWWSSSS': w5.south.south.south.south,
        # 10 moves away
        'SSSSSSSSS': s10, 'NNNNNNNNN': n10, 'WWWWWWWWW': w10, 'EEEEEEEEE': e10,
        'WWWWWWWWWN': w9.north, 'WWWWWWWWWS': w9.south, 'EEEEEEEEEN': e9.north, 'EEEEEEEEES': e9.south,
        'SSSSSSSSSE': s9.east, 'SSSSSSSSSW': s9.west, 'NNNNNNNNNW': n9.west, 'NNNNNNNNNE': n9.east,
        'NNNNNNNNEE': n8.east.east, 'SSSSSSSSEE': s8.east.east, 'NNNNNNNNWW': n8.west.west, 'SSSSSSSSWW': s8.west.west,
        'EEEEEEEENN': e8.north.north, 'EEEEEEEESS': e8.south.south, 'WWWWWWWWNN': w8.north.north, 'WWWWWWWWNN': w8.north.north,
        'WWWWWWWNNN': w7.north.north, 'WWWWWWWSSS': w7.south.south, 'EEEEEEESSS': e7.south.south.south, 'EEEEEEEWWW': e7.north.north.north,
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
    if not(len(board.current_player.ships) == 0 and board.current_player.halite < 500):
        log(str(step) + '|-----------------------------------------------------------------------')

    for ship_id in ships:
        if ship_id in board.current_player.ship_ids:
            log(' Pos:' + str(board.ships[ship_id].position) + ', cargo: ' + str(board.ships[ship_id].halite) + ', player halite: ' + str(board.current_player.halite))
                
            next_action, action_type = DecisionShip(board, ship_id, step).determine()
                
            if action_type != 'mine':
                actions[ship_id] = movement_dictionary[action_type]
                board.ships[ship_id].next_action = next_action
                if step == 200: log(board)
                board = board.next()
                if step == 200: log(board)
        # else:
        #     log(' Not found')

    shipyard_ids = ShipyardDecisions(board, board.current_player, step).determine()

    for shipyard_id in board.current_player.shipyard_ids:
        if shipyard_id in shipyard_ids:
            actions[shipyard_id] = 'SPAWN'
            board.shipyards[shipyard_id].next_action = ShipyardAction.SPAWN
            
            board = board.next()
        
    return actions
