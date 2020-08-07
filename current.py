from kaggle_environments.envs.halite.helpers import Point, Board, ShipAction, ShipyardAction
import pandas as pd


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

    def __init__(self, board: Board, ship):
        # Given values
        self.board = board
        self.ship = ship

        # Some usefull properties
        self.player = self.board.current_player  # Player
        self.ship_cargo = ship.halite
        self.current_cell = ship.cell
        self.current_position = ship.position
        self.step = board.observation['step'] + 1

        # All moves ship can take
        self.moves = {"N": ShipAction.NORTH, 'S': ShipAction.SOUTH, 'W': ShipAction.WEST,
                      'E': ShipAction.EAST, 'convert': ShipAction.CONVERT, 'mine': None}
        # The set of moves that should not be taken
        self.eliminated_moves = ['None']

        # Weights of different moves
        self.weights = {"N": 0, "E": 0, "W": 0, "S": 0, "mine": 0, "convert": 0, 'None': 0}

        # The object's relative situation to other ship/shipyards
        self.locator = Locator(board, ship)
        self.Ships = self.locator.get_ship_info()
        self.Shipyards = self.locator.get_shipyard_info()
        self.grid = self.locator.generate_grid_df()

        # Default move which is set to mining (None)
        self.next_move = None

        # This variable holds the direction that is being evaluated
        self.current_direction = ""

    def determine(self):
        """ Returns the next action decided for the ship based on the observations that have been made. """
        # Get the weights for main four directions
        self.weight_moves()
        self.round()  # Round the weights
        self.apply_elimination()  # Apply the eliminations

        # Sort the values
        sorted_weights = {k: v for k, v in sorted(self.weights.items(), key=lambda item: item[1], reverse=True)}

        log('  weights: ' + str(sorted_weights))

        # Choose the action with highest value given that it is not eliminated
        for action in sorted_weights.keys():
            log('  ## next_move: ' + action)
            return self.moves[action]

        # If none were chosen, then just return the default move which is mining
        return self.next_move

    def weight_moves(self):
        """ 
            This is the main function and runs other helper functions within the module to in order to weight
            the different moves that could be taken.

            The weightings will take numerous considerations into account (halite, closeness to shipyard, etc.)
            yet there are going to be some values manual multiplications in order to encourage or discourage certain
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
        # list of all directions in 5 moves apart
        dirs = list(self.grid.columns)
        # Iterate through different directions
        for direction in dirs:
            # Set the global direction to the one at hand
            self.current_direction = direction

            # If there was a ship
            if not pd.isna(self.grid[direction].ship_id):
                # If it was my ship
                if self.grid[direction].my_ship == 1:
                    if self.Ships[self.grid[direction].ship_id]['moves'] == 1:
                        self.eliminated_moves.append(direction)  # Just don't go there
                    else:
                        # Distribute the ships so they won't end up in the same place
                        self.avoid_self_colision(self.grid[direction].ship_id)
                else:
                    # If it was not my ship
                    if self.Ships[self.grid[direction].ship_id]['moves'] == 1 and \
                            self.Ships[self.grid[direction].ship_id]['cargo'] < self.ship_cargo:
                        # If there was an enemy ship one move away with less halite then eliminate the move and don't mine
                        log('  Enemy ship at ' + direction + ', No mining!')
                        self.eliminated_moves.append(direction)
                        self.eliminated_moves.append('mine')
                        # Go to the closest shipyard preferably
                        self.go_to_closest_shipyard(self.ship_cargo ** 2);
                    else:
                        self.deal_enemy_ship(self.grid[direction].ship_id)

            # If there was a shipyard
            if not pd.isna(self.grid[direction].shipyard_id):
                # If it was my shipyard
                if self.grid[direction].my_shipyard == 1:
                    self.deposit()
                else:
                    # If it was not my shipyard
                    self.attack_enemy_shipyard(self.grid[direction].shipyard_id)

            # Consider the main four direction solely based on the halite amount
            for main_dir in ["N", "S", "W", "E"]:
                if main_dir in direction:
                    moves_apart: float = direction.count(main_dir)
                    main_dir_encourage: float = self.grid[direction].halite * (21 - moves_apart) / (moves_apart * 21)
                    # log('    Adding direction: ' + main_dir + ', moves apart:' + str(moves_apart) + ', enc: ' + str(round(main_dir_encourage,3)))
                    self.weights[main_dir] += main_dir_encourage

            # In order to keep the mining as an option as well: the mining option will get add to only if amount of
            # grid in the current cell is higher than other places
            mining_trigger = round(
                (self.current_cell.halite - self.grid[direction].halite) / self.grid[direction].moves, 3)
            # log('    At the end of the directions adding ' +  str(mining_trigger) + ' to the mining weight.')
            self.weights['mine'] += mining_trigger

        # The correlation of the mining with cell's halite
        mining_encouragement = self.current_cell.halite
        self.weights['mine'] += mining_encouragement ** 2
        log('  Mining-enc: ' + str(mining_encouragement))

        # Weight the CONVERT option
        self.weight_convert()
        # self.shipyard_status()

    def add_accordingly(self, value):
        """ Adds a value to a weight according to the relative distance. """
        # Get the repetition of each one letter direction
        rep_X: int = self.grid[self.current_direction]['movesX']
        rep_Y: int = self.grid[self.current_direction]['movesY']
        dirX = self.grid[self.current_direction]['dirX']
        dirY = self.grid[self.current_direction]['dirY']

        weightX = self.grid[self.current_direction]['weightX']
        weightY = self.grid[self.current_direction]['weightY']

        if value !=0: log('    dir:' +self.current_direction + ', adding ' + str(round(value, 3)) + ' to ' + dirX + ', moves: ' + str(
            rep_X) + ' and ' + dirY + ", moves: " + str(rep_Y))

        if weightX != 0: self.weights[dirX] += value * weightX
        if weightY != 0: self.weights[dirY] += value * weightY

    def weight_convert(self, threshold=2000):
        """ Weights the option for ship convertion. """
        # 1. If they are no shipyards left
        no_yards_left = len(self.player.shipyards) == 0
        # 2. If it is the end of the game and we have more than 500 halite in our cargo
        end_of_game_conversion = (self.step > 396 or self.near_end()) and self.ship.halite >= 500
        # 3. There will be a threshold for the amount of cargo any ship could have
        threshhold_reach = self.ship.halite > threshold
        # 4. On shipyard already
        on_shipyard = self.ship.cell.shipyard is not None

        # 5. It would not make sense to have a shipyard four moves away from another ship
        shipyard_id = self.closest_shipyard()
        if shipyard_id != 0:
            shipyard_distance = self.Shipyards[shipyard_id].moves > 3
        else:
            shipyard_distance = True

        if (no_yards_left or end_of_game_conversion or threshhold_reach) and not on_shipyard and shipyard_distance:
            self.weights['convert'] = 3000 * (self.step // 10 + 1)
        elif not on_shipyard and shipyard_distance:
            self.weights['convert'] = round((self.ship_cargo - 1000) / (self.step // 50 + 1), 3)
        # else:
        #     self.weights['convert'] = round((self.ship_cargo - threshold) / (self.step // 50 + 1), 3)

    def deposit(self):
        """ Weights the tendency to deposit and adds to the directions which lead to the given shipyard. """
        log('  Deposit:') # logging
        # measure tendency
        deposit_tendency = 10 * self.ship_cargo
        self.add_accordingly(deposit_tendency)

    def attack_enemy_shipyard(self, shipyard_id):
        """ Weights the tendency to attack the enemy shipyard. """
        # Get the shipyard's info
        shipyard = self.board.shipyards[shipyard_id]
        num_enemy_shipyards = len(shipyard.player.shipyards)

        log('  Attacking enemy shipyard') # Logging

        attacking_enemy_shipyard_tendency = (shipyard.player.halite + 0.5) / (num_enemy_shipyards + 0.5)

        self.add_accordingly(attacking_enemy_shipyard_tendency)

    def deal_enemy_ship(self, ship_id):
        """ 
            This function will evaluate to either attack or get_away from an enemy ship based on 
            the simple observation: If my ship had more cargo then I should not attack. 
        """
        # The amount of cargo caried by the opponent's ship
        oppCargo = self.Ships[ship_id].cargo
        log(' Dealing with enemy ship:') # Logging
        # If the ship's cargo was more than enemy's cargo and it was not equal to zero then get away
        if self.ship_cargo >= oppCargo and self.ship_cargo != 0:
            # If my ship's cargo was not less than enemy's
            self.get_away()
        else:
            # If the ship had more halite then attack
            self.attack_enemy_ship(oppCargo - self.ship_cargo)

    def attack_enemy_ship(self, diff):
        """ This function encourages attacking the enemy ship. """
        # Get the ship's info
        log('  Attacking enemy ship')
        attack_encouragement = 10 * diff

        self.add_accordingly(attack_encouragement)

    def get_away(self):
        """ 
            This function is called when my ship has a possible change of getting followed by 
            another enemy ship with lower halite.
            
            There can multiple options that given the parameters and the manual weightings one 
            could have a higher value and hence tendency.
        """
        log('  Get away') # Logging

        direction_discouragement = -10 * self.ship_cargo
        self.add_accordingly(direction_discouragement)

        # The weights for the closest shipyard goes up.
        closest_shipyard_encouragement = 10 * self.ship_cargo 
        self.go_to_closest_shipyard(closest_shipyard_encouragement)

    def go_to_closest_shipyard(self, value):
        """ Encourage movement towards the nearest shipyard """
        # Get the id of the closest shipyard to the ship
        closest_shipyard_id = self.closest_shipyard()
        
        # If there where any shipyards available then increase the tendency to get to them
        if closest_shipyard_id != 0:
            log('   closest_id: ' + closest_shipyard_id)
            dirX, dirY = self.Shipyards[closest_shipyard_id]['dirX'], self.Shipyards[closest_shipyard_id]['dirY']
            log('   adding ' + value + ' to ' + dirX + ' and ' + dirY)

            self.weights[dirX] += value
            self.weights[dirY] += value

    # def shipyard_status(self):
    #     """ This function gives tendency to go to shipyards within the map. """
    #     if not self.Shipyards.empty:
    #         for shipyard in self.player.shipyards:
    #             self.analyze_shipyard_surroundings(shipyard.id)

    # def analyze_shipyard_surroundings(self, shipyard_id):
    #     """ Checks to see if a given shipyard needs protection or not? """
    #     shipyard = self.board.shipyards[shipyard_id]

    #     log('  Shipyard status:' + shipyard.id)

    #     shipyard_locator = Locator(self.board, shipyard)
    #     shipyard_grid = shipyard_locator.generate_grid_df()

    #     for direction in list(shipyard_grid.columns):
    #         ship_id = shipyard_grid[direction].ship_id
    #         # IF there is a ship on that cell
    #         if not pd.isna(ship_id):
    #             dirX, dirY = determine_directions(self.current_position, shipyard.position)

    #             if shipyard_grid[direction].my_ship == 1:
    #                 my_ship_weight = -2 * self.ship_cargo
    #                 log('    my ship, adding')
    #                 self.add_accordingly(dirX, dirY, my_ship_weight)
    #             else:
    #                 enemy_ship_weight = 3 * self.ship_cargo
    #                 log('    enemy ship, adding')
    #                 self.add_accordingly(dirX, dirY, enemy_ship_weight)

    def closest_shipyard(self):
        """ This function will return the distance to the closest shipyard's id. """
        shipyard_id = 0  # The default value would be zero meaning that they either no shipyard or I did not have any
        # First we should check to see if there are any Shipyards at all
        if not self.Shipyards.empty:
            # Then we should check to see if I have any shipyards
            if not self.Shipyards.T[self.Shipyards.T['my_shipyard'] == 1].empty:
                min_val = self.Shipyards.T[self.Shipyards.T['my_shipyard'] == 1]['moves']
                tMyShipyards = self.Shipyards.T[self.Shipyards.T['my_shipyard'] == 1]
                # log('min-value: ' + str(min_val) + ' ' + str(tMyShipyards))
                shipyard_id = list(tMyShipyards[tMyShipyards['moves'] == min_val].T.columns)[0]

        return shipyard_id

    def avoid_self_colision(self, ship_id):
        """ This function is called to avoid lower the tendency for cell with my own ships. """
        log('  Avoid Collision')

        # Getting ship's info
        oppShipCargo = self.Ships[ship_id].cargo
        diff = abs(self.ship_cargo - oppShipCargo)

        # Given that they are not one move apart the discouragement should not be that strong
        discourage_collision = -1 * (diff + 3)
        
        self.add_accordingly(discourage_collision)

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

    def apply_elimination(self):
        """ Eliminates the moves to be eliminated. """
        for move in self.eliminated_moves:
            log('  Eliminating ' + move)
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
    def __init__(self, board: Board, shipyards):
        """
            Decides the Shipyard's next action based on the given parameters
            board: The board that we will be observing
            shipyards: All the shipyards that the player has
        """
        self.board = board
        self.Shipyards = shipyards

    def determine(self):
        pass

    def weight_shipyard_tendencies(self):
        for shipyard in self.Shipyards:
            pass


class Locator:
    """ This object returns dataframes containing the information about other ships/shipyards on the map and also in the given ship's grid. """
    def __init__(self, board, ship):
        self.board = board
        self.ship = ship
        self.ship_position = ship.position
        # Get the grid
        self.grid = grid(ship.cell)

    def get_ship_info(self):
        """ Returns the info about ships in all of the board. """
        ships_info = {}

        for ship_id, ship in self.board.ships.items():
            base_info = {"my_ship": 0, "moves": 0, "position": (ship.position.x, ship.position.y),
                         'cargo': ship.cell.halite, 'moves': count_moves(self.ship.position, ship.position),
                         'dirX': (determine_directions(self.ship_position, ship.position))[0],
                         'dirY': (determine_directions(self.ship_position, ship.position))[1]}

            if ship_id in self.board.current_player.ship_ids and ship.id != self.ship.id:
                base_info['my_ship'] = 1
                ships_info[ship_id] = base_info
            elif not (ship_id in self.board.current_player.ship_ids):
                ships_info[ship_id] = base_info

        return pd.DataFrame(ships_info)

    def get_shipyard_info(self):
        """ Returns the info about shipyards in all of the board. """
        shipyards_info = {}

        for shipyard_id, shipyard in self.board.shipyards.items():
            base_info = {"my_shipyard": 0,
                         "position": (shipyard.position.x, shipyard.position.y),
                         'dirX': (determine_directions(self.ship_position, shipyard.position))[0],
                         'dirY': (determine_directions(self.ship_position, shipyard.position))[1],
                         'player_halite': shipyard.player.halite,
                         'moves': count_moves(self.ship.position, shipyard.position)}

            if shipyard_id in self.board.current_player.shipyard_ids:
                base_info['my_shipyard'] = 1
                shipyards_info[shipyard_id] = base_info
            else:
                shipyards_info[shipyard_id] = base_info

        return pd.DataFrame(shipyards_info)

    def generate_grid_df(self):
        """ Generates a Dataframe describing the information of objects and cells in the 5x5 grid of the ship. """
        all_dirs = {}

        total_moves = len(self.grid) / 4

        for direction, cell in self.grid.items():
            weight = (total_moves - len(direction)) / (total_moves * len(direction)) 

            base_info = {
                "ship_id": None, "shipyard_id": None,
                "my_ship": 0, "my_shipyard": 0,
                "halite": 0, "moves": 0,
                "movesX": 0, "movesY": 0,
                "dirY": 'None', "dirX": 'None',
                'weight': weight
            }

            if "N" in  direction:
                base_info['dirY'] = 'N'
                base_info['movesY'] = direction.count("N")
            elif "S" in  direction:
                base_info['dirY'] = 'S'
                base_info['movesY'] = direction.count("S")

            if "W" in  direction:
                base_info['dirX'] = 'W'
                base_info['movesX'] = direction.count("W")
            elif "E" in  direction:
                base_info['dirX'] = 'E'
                base_info['movesX'] = direction.count("E")

            if cell.ship is not None:
                base_info["ship_id"] = cell.ship.id
                if cell.ship.id in self.ship.player.ship_ids:
                    base_info["my_ship"] = 1

            if cell.shipyard is not None:
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
import random


def determine_directions(point1, point2, size=21):
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
    best_x, best_y = 'None', 'None'

    if diff_x_1 > diff_x_2:
        if x2 - x1 > 0:
            best_x = "W"
        elif x2 - x1 < 0:
            best_x = "E"
    else:
        if x2 - x1 > 0:
            best_x = "E"
        elif x2 - x1 < 0:
            best_x = "W"

    if diff_y_1 > diff_y_2:
        if y2 - y1 > 0:
            best_y = "S"
        elif y2 - y1 < 0:
            best_y = "N"
    else:
        if y2 - y1 > 0:
            best_y = "N"
        elif y2 - y1 < 0:
            best_y = "S"

    return best_x, best_y


def grid(cell):
    """ Returns a dictionary of cells which are in 4 moves distance of the given cell """
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
        'EESSS':s3.east.east, 'WWSSS': s3.west.west,'EENNN': n3.east.east, 'WWNNN': n3.west.west,
        'EEESS': e3.south.south, 'EEENN': e3.north.north, 'WWWSS': w3.south.south, 'WWWNN': w3.north.north,
        # 6 moves away
        'SSSSSS': s6, 'NNNNNN': n6, 'WWWWWW': w6, 'EEEEEE': e6,
        'WWWWWN': w5.north, 'WWWWWS': w5.south, 'EEEEEN': e5.north, 'EEEEES': e5.south,
        'SSSSSE': s5.east, 'SSSSSW': s5.west, 'NNNNNW': n5.west, 'NNNNNE': n5.east,
        'WWWWNN': w4.north.north, 'WWWWSS': w4.south.south, 'EEEENN': e4.north.north, 'EEEESS': e4.south.south,
        'EEENNN': e3.north.north.north, 'EEESSS': e3.south.south.south, 'WWWNNN': w3.north.north.north, 'WWWSSS': w3.south.south.south,
        'NNNNEE': n4.east.east, 'NNNNWW': n4.west.west, 'SSSSWW': s4.west.west, 'SSSSEE': s4.east.east
    }


def count_moves(point1, point2, size=21):
    """ 
        Returns the minimum number of between moves to go from point1 to point2.
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


log('logs:', 0)


def agent(obs, config):
    # Make the board
    board = Board(obs, config)

    # Step of the board
    step = board.observation['step']
    # Current player info
    me = board.current_player  # Player Object

    new_board = Board(obs, config)
    log('-----------------------------------------------------------------')
    log(step + 1)
    for ship in me.ships:
        log('ship-id:' + ship.id + ', pos:' + str(ship.position) + ', cargo: ' + str(ship.halite) + ', player halite: ' + str(me.halite))

        if ship.id in new_board.ships.keys():
            current_ship = new_board.ships[ship.id]
            decider = DecisionShip(new_board, current_ship)
            ship.next_action = decider.determine()

            new_board = board.next()
        log('#########')
    for shipyard in me.shipyards:
        # If there were no ships on the yard
        if new_board.shipyards[shipyard.id].cell.ship is None and step < 392:
            if len(me.ships) == 0:
                shipyard.next_action = ShipyardAction.SPAWN

            if step % 3 == 1:
                shipyard.next_action = ShipyardAction.SPAWN

            if step > 200 and me.halite > 10000 + len(me.ships) * 1000:
                shipyard.next_action = ShipyardAction.SPAWN

        new_board = board.next()

    log(me.next_actions)
    return me.next_actions
