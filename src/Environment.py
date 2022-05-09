from Agent import Agent
from Enemy import Enemy
import random
import copy
import matplotlib.pyplot as plt
import numpy as np


class Environment:

    def __init__(self):

        self.length_grid = 25
        self.grid = []
        self.obstacle_positions_list = [
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0,5), (0, 19), (0, 20), (0, 21), (0, 22), (0, 23), (0, 24), (1,0), (1,24), (2,0), (2,3), (2,4), (2,8 ), (2,9 ), (2,11 ), (2,12 ), (2,13 ),
            (2,15 ), (2,16 ), (2,20), (2,21 ), (2,24 ), (3,0 ), (3,2 ), (3,3 ), (3,4 ), (3,8 ), (3,9 ), (3,11 ), (3,13 ), (3,15 ), (3,16 ), (3,20 ), (3,21 ), (3,22 ), (3,24 ), (4,0 ), (4,2 ),
            (4,3 ), (4,4 ), (4,20 ), (4,21 ), (4,22 ), (4,24 ), (5,0 ), (5,24), (8,2 ), (8,3 ), (8,8 ), (8,9 ), (8,10 ), (8,11 ), (8,13 ), (8,14 ), (8,15 ), (8,16 ), (8,21 ), (8,22 ), (9,2 ),
            (9,3 ), (9,8 ), (9,9 ), (9,10 ), (9,14 ), (9,15 ), (9,16 ), (9,21 ), (9,22 ), (10,8 ), (10,9 ), (10,15 ), (10,16), (11,2 ), (11,3 ), (11,8 ), (11,16 ), (11,21 ), (11,22 ), (12,2 ),
            (12,12 ), (12,22 ), (13,2 ), (13,3 ), (13,8 ), (13,16 ), (13,21 ), (13,22 ), (14,8 ), (14,9 ), (14,15 ), (14,16 ), (15,2 ), (15,3 ), (15,8 ), (15,9), (15,10 ), (15,14 ), (15,15 ),
            (15,16 ), (15,21 ), (15,22 ), (16,2 ), (16,3 ), (16,8 ), (16,9 ), (16,10 ), (16,11 ), (16,13 ), (16,14 ), (16,15 ), (16,16 ), (16,21 ), (16,22 ), (19,0 ), (19,24), (20,0 ), (20,2 ),
            (20,3 ), (20,4 ), (20,20 ), (20,21 ), (20,22 ), (20,24 ), (21,0 ), (21,2 ), (21,3 ), (21,4 ), (21,8 ), (21,9 ), (21,11 ), (21,13 ), (21,15 ), (21,16 ), (21,20 ), (21,21 ), (21,22 ),
            (21,24), (22, 0), (22, 3), (22, 4), (22, 8), (22, 9), (22, 11), (22, 12), (22, 13), (22, 15), (22, 16), (22, 20 ), (22, 21), (22, 24 ), (23,0 ), (23,24), (24, 0), (24, 1), (24, 2),
            (24, 3), (24, 4), (24, 5), (24, 19), (24, 20), (24, 21), (24, 22), (24, 23), (24, 24)]

        self.agent = Agent()
        self.agent_move_back = False
        self.agent_collided = False
        self.nb_collision = 0
        self.list_mean_collision = []
        self.nb_death = 0
        self.list_mean_death = []

        self.number_of_enemies = 4
        self.enemies_list = []
        for id in range(self.number_of_enemies):
            self.enemies_list.append(Enemy(id + 1))
        self.enemy_positions_list = [(1, 12), (6, 6), (6, 12), (6, 18)]

        self.number_of_food = 15
        self.food_symbol = "$"
        self.list_mean_food_collected = []
        self.nb_collected_food = 0

        # Defined in section 5.1 "Experimental design" page 18
        self.number_training_env = 300
        self.number_test_env = 50
        self.number_of_training = 20


        # Values defined for Q-CON, see table 1 page 19
        one_over_T_min = 20
        one_over_T_max = 60
        self.one_over_temperature_coeff_list = np.linspace(one_over_T_min,
                                                            one_over_T_max,
                                                            int(self.number_training_env / self.number_of_training)).tolist()
        self.one_over_temperature_coeff_list = [round(elem, 2) for elem in self.one_over_temperature_coeff_list]

        self.training_mode = True
        self.verbose = False

        self.reset = False

        self.initialize_grid()

    def initialize_grid(self):
        self.reset = False

        # Create empty 25*25 grid
        self.grid = []
        for i in range(self.length_grid):
            self.grid.append([])
            for j in range(self.length_grid):
                self.grid[i].append([" "])

        # Add agent
        self.agent.set_position((18, 12))
        self.grid[18][12].append(self.agent.get_symbol())
        self.agent_move_back = False
        self.agent.old_position = self.agent.get_position()
        self.agent.set_last_action([0, 0, 0, 0])
        self.agent.set_energy(self.agent.energy_init)
        self.agent_collided = False

        # Add enemies
        for i in range(self.number_of_enemies):
            pos_x, pos_y = self.enemy_positions_list[i] # compute position and place the enemy in the grid
            self.enemies_list[i].set_position((pos_x, pos_y))
            self.grid[pos_x][pos_y].append(self.enemies_list[i].get_symbol()) 

        # Add obstacles
        for o in self.obstacle_positions_list:
            self.grid[o[0]][o[1]].append("O")

        # Add food
        self.nb_collected_food = 0
        self.number_of_collected_foods = 0
        self.food_location = []
        empty_cells = self.get_list_empty_cells()
        for i in range (self.number_of_food):
            pos = empty_cells.pop(random.randrange(len(empty_cells)))
            self.food_location.append(pos)
            self.grid[pos[0]][pos[1]].append(self.food_symbol)

    def get_list_empty_cells(self):
        """
        Compute the list of empty cells
        Used for placing foods in initialize_grid()

        Returns:
            List<Tuple>: List of empty cells coordinates
        """
        empty_cells = []
        for i in range(len(self.grid[0])):
            for j in range(len(self.grid[1])):
                if len(self.grid[i][j]) == 1:
                    empty_cells.append((i,j))
        return empty_cells


    def run(self):
        """
        Run an iteration of the simulation, agent and enemies move once, environnement updated according to the movements.
        """
        self.observation()
        action = self.agent.choose_action(self.training_mode)
        new_pos_content = self.move(action, self.agent)

        if self.agent_collided == True:
            self.nb_collision += 1

        if "E" in new_pos_content:
            self.update_agent(new_pos_content)
            if self.verbose:
                print("Simulation finished : Agent killed by enemy")
            self.reset = True
            return
        else:
            # Enemies' movement
            for e in self.enemies_list:
                surroundings = self.get_surroundings(e)
                action = e.choose_action(self.agent.get_position(), surroundings)
                self.move(action, e)

        self.update_agent(new_pos_content)

        if self.verbose:
            print("----------------------------")
            print('\n'.join(''.join(str(x) for x in row) for row in self.grid))
            print("Energie restante : ", self.agent.get_energy())

        if self.reset == True:
            if self.verbose:
                print("Simulation finished : Agent killed by enemy")
            return

        if self.agent.get_energy() <= 0:
            if self.verbose:
                print("Simulation finished : Agent out of energy")
            self.reset = True
            return

        if self.nb_collected_food >= self.number_of_food:
            if self.verbose:
                print("Simulation finished : All food collected")
            self.reset = True
            return

    def move(self, action: str, individual):
        """[summary]

        Args:
            action (str): direction of the movement
            individual (Agent or Enemy): individual that is moving

        Returns:
            [List<String>]: List of element contained in the new position
        """
        if type(individual) == Enemy and action == "Stay":
            return

        elif type(individual) == Agent:
            individual.consume_energy()

        pos_x, pos_y = individual.get_position()
        new_pos = None

        if action == "North":
            if type(individual) == Agent:
                self.agent.set_last_action([1, 0, 0, 0])
            # Check if there is an obstacle or out of grid
            if pos_x == 0 or "O" in self.grid[pos_x - 1][pos_y]:
                if type(individual) == Agent: # If it's an agent mouvement, set collision bool to true
                    self.agent_collided = True
                if self.verbose:
                    print("Mouvement impossible : Obstacle présent ou sortie de grille")
                return self.grid[pos_x][pos_y]
            # Compute next pos
            else:
                new_pos = (pos_x - 1, pos_y)

        elif action == "South":
            if type(individual) == Agent:
                self.agent.set_last_action([0, 0, 1, 0])
            # Check if there is an obstacle or out of grid
            if pos_x == self.length_grid - 1 or "O" in self.grid[pos_x + 1][pos_y]:
                if type(individual) == Agent: # If it's an agent mouvement, set collision bool to true
                    self.agent_collided = True
                if self.verbose:
                    print("Mouvement impossible : Obstacle présent ou sortie de grille")
                return self.grid[pos_x][pos_y]
            # Compute next pos
            else:
                new_pos = (pos_x + 1, pos_y)

        elif action == "East":
            if type(individual) == Agent:
                self.agent.set_last_action([0, 1, 0, 0])
            # Check if there is an obstacle or out of grid
            if pos_y == self.length_grid - 1 or "O" in self.grid[pos_x][pos_y + 1]:
                if type(individual) == Agent: # If it's an agent mouvement, set collision bool to true
                    self.agent_collided = True
                if self.verbose:
                    print("Mouvement impossible : Obstacle présent ou sortie de grille")
                return self.grid[pos_x][pos_y]
            # Compute next pos
            else:
                new_pos = (pos_x, pos_y + 1)

        elif action == "West":
            if type(individual) == Agent:
                self.agent.set_last_action([0, 0, 0, 1])
            # Check if there is an obstacle or out of grid
            if pos_y == 0 or "O" in self.grid[pos_x][pos_y - 1]:
                if type(individual) == Agent: # If it's an agent mouvement, set bump bool to true
                    self.agent_collided = True
                if self.verbose:
                    print("Mouvement impossible : Obstacle présent ou sortie de grille")
                return self.grid[pos_x][pos_y]
            # Compute next pos
            else:
                new_pos = (pos_x, pos_y - 1)
        else:
            print("Environnement move() : action not recognized, should not happen")

        # Move the individual
        if action != "Stay":
            if type(individual) == Agent:
                self.agent_collided = False
                old_pos = self.agent.get_position()

            self.grid[pos_x][pos_y].remove(individual.get_symbol())
            individual.set_position(new_pos)

            if type(individual) == Agent:
                if self.agent.old_position == self.agent.get_position():
                    self.agent_move_back = True
                else:
                    self.agent_move_back = False
                self.agent.old_position = old_pos

        new_pos_x, new_pos_y = new_pos
        new_pos_content = self.grid[new_pos_x][new_pos_y]
        if self.verbose:
            if type(individual) == Agent:
                print("Position and element before:", (new_pos_x, new_pos_y), new_pos_content)

        self.grid[new_pos_x][new_pos_y].append(individual.get_symbol())

        return new_pos_content

    def update_agent(self, new_pos_content: str):
        """
        Update the agents observation and backpropagate errore
        Args:
            new_pos_content (str): Content of the cell the agent is moving to : enemy, food, empty, itself (if the agent didn't move)
        """
        reward = self.reward_feedback(new_pos_content)
        if self.training_mode:
            self.observation() # Update sensors' observation
            self.agent.backpropagate_reward_error(reward) # Backpropagate error
        self.update_status(reward)

    def reward_feedback(self, new_pos_content: str):
        """
        Compute reward from the new state
        Args:
            new_pos_content (str): Content of the cell the agent is moving to : enemy, food, empty, itself (if the agent didn't move)
        Returns:
            [float]: obtained reward, as defined page 14
        """
        for e in self.enemies_list:
            if self.agent.get_position() == e.get_position():
                return -1

        if "$" in new_pos_content:
            self.agent.set_energy(self.agent.get_energy() + 15)
            return 0.4
    
        elif "O" in new_pos_content:
            print("Environnement reward_feedback : Agent in the same position with a wall, should not happend")
            return None

        # Reward from Article 2, page 4
        elif self.agent_collided == True:
            return -0.05

        elif self.agent_move_back == True:
            return -0.05

        return 0

    def update_status(self, reward: float):
        """
        Update environment according to the perceived reward
        Args:
            reward (float): Obtained reward
        """
        if reward == 0.4:
            self.nb_collected_food += 1
            pos_x, pos_y = self.agent.get_position()
            self.food_location.remove((pos_x, pos_y))
            self.grid[pos_x][pos_y].remove("$")

        elif reward == -1:
            self.nb_death += 1
            self.reset = True

    def observation(self):
        """
        Update every sensors
        """
        # Reinitialize the agent's sensors inputs
        self.agent.init_sensor_value()

        for sensor_type in self.agent.get_sensors_id():
            if sensor_type == "Y":
                self.update_sensors(sensor_type, "$")
            elif sensor_type == "O" or sensor_type == "X":
                self.update_sensors(sensor_type, "$")
                self.update_sensors(sensor_type, "E")
            elif sensor_type == "o":
                self.update_sensors(sensor_type, "O")

        # Update the agent's energy input
        self.update_energy_input(self.agent.get_energy())

        # Update the agent's shock input
        self.agent.set_collide(self.agent_collided)

    def update_sensors(self, sensor_type: str, element_to_check: str):
        """
        Update a sensor

        Args:
            sensor_type (str): Type of sensor ("X", "O", "Y" "o")
            element_to_check (str): Element to check ("$", "E", "O")
        """
        list_sensors_position = None
        list_sensors_scope = None
        list_updated_sensors = None

        if sensor_type == "Y":
            list_sensors_position = self.agent.get_Y_position()
            list_sensors_scope = self.agent.get_Y_scope()
            if element_to_check == "$":
                list_updated_sensors = self.agent.get_food_Y_values()

        elif sensor_type == "O":
            list_sensors_position = self.agent.get_O_position()
            list_sensors_scope = self.agent.get_O_scope()
            if element_to_check == "$":
                list_updated_sensors = self.agent.get_food_O_values()
            elif element_to_check == "E":
                list_updated_sensors = self.agent.get_enemy_O_values()

        elif sensor_type == "X":
            list_sensors_position = self.agent.get_X_position()
            list_sensors_scope = self.agent.get_X_scope()
            if element_to_check == "$":
                list_updated_sensors = self.agent.get_food_X_values()
            elif element_to_check == "E":
                list_updated_sensors = self.agent.get_enemy_X_values()

        elif sensor_type == "o":
            list_sensors_position = self.agent.get_o_position()
            list_sensors_scope = self.agent.get_o_scope()
            if element_to_check == "O":
                list_updated_sensors = self.agent.get_obstacle_o_values()

        for i in range(len(list_sensors_position)):
            sensor_position = list_sensors_position[i] # relative position of the sensor to the agent
            (sensor_x_pos, sensor_y_pos) = tuple(map(sum, zip(self.agent.get_position(), sensor_position))) # absolute position of the sensor to examine

            for j in range(len(list_sensors_scope)):
                sensor_scope = list_sensors_scope[j] # relative scope position to the origin (0, 0)
                (sensor_x_scope, sensor_y_scope) = tuple(map(sum, zip((sensor_x_pos, sensor_y_pos), sensor_scope))) # absolute positions of the cell to examine

                # Case of examined cell being out of bounds
                if sensor_x_scope < 0 or self.length_grid <= sensor_x_scope or sensor_y_scope < 0 or self.length_grid <= sensor_y_scope:
                    if sensor_type == "o":
                        list_updated_sensors[i] = 1
                    continue

                if element_to_check in self.grid[sensor_x_scope][sensor_y_scope]:
                    list_updated_sensors[i] = 1
                    break

    def update_energy_input(self, agent_energy):
        """
        Update the energy level of the agent for the network input

        Args:
            agent_energy (Integer): energy of the agent
        """
        energy_level_input = [0 for _ in range(16)]
        step_level = self.agent.get_max_energy() / 16
        if agent_energy == self.agent.get_max_energy():
            energy_level_input[-1] = 1
        else:
            energy_level_input[int(agent_energy // step_level)] = 1
        self.agent.set_input_energy(energy_level_input)

    def get_surroundings(self, enemy: Enemy):
        """
        Compute the observation of the enemy on its 4 adjacent cells

        Args:
            enemy (Enemy): the enemy whose surrounding we are computing

        Returns:
            List<String>: Content of adjacent cells in following order : North, South, East, West
        """
        pos_x, pos_y = enemy.get_position()
        if pos_x <= 0:
            north = "O"
        else:
            north = self.grid[pos_x - 1][pos_y]
        if pos_x >= self.length_grid - 1:
            south = "O"
        else:
            south = self.grid[pos_x + 1][pos_y]
        if pos_y >= self.length_grid - 1:
            east = "O"
        else:
            east = self.grid[pos_x][pos_y + 1]
        if pos_y <= 0:
            west = "O"
        else:
            west = self.grid[pos_x][pos_y - 1]

        return [north, south, east, west]

    def experiment(self, save_file="Resultats/res_experiment.txt"):
        """
        Execute experiments as defined in the article. See section 5.1 "Experimental design" page 18
        Run 300 training iteration in which the action choosen by the agent is done by a stochastique action selector (Exploration / Exploitation).
        Every 20 training iteration done, run 50 test iterations in which the action taken by the agent is choosen by selecting the max Q-Value (Exploitation only).

        Args:
            save_file (str, optional): Path to file in which results are saved. Defaults to "../Resultats/res_experiment.txt".
        """
        training_serie_nb = -1

        for train_nb in range(self.number_training_env):
            if train_nb % self.number_of_training == 0:
                print("---------- Test after {} trainings ---------- ".format(train_nb))
                self.test_iterations()
                training_serie_nb += 1
                self.agent.temperature = 1 / self.one_over_temperature_coeff_list[training_serie_nb]

            self.initialize_grid()
            while (self.reset == False):
                self.run()

        print("---------- Test after 300 trainings ----------")
        self.test_iterations()

        x_values = list(range(0, self.number_training_env + self.number_of_training, self.number_of_training))
        print("nb_training", len(x_values), x_values)
        print("mean_collected_food", len(self.list_mean_food_collected), self.list_mean_food_collected)
        print("mean_collision", self.list_mean_collision)

        # Save the results:
        fic = open(save_file, "a")
        fic.write("nb_training: {}\n".format(x_values))
        fic.write("mean_collected_food: {}\n".format(self.list_mean_food_collected))
        fic.write("mean_collision: {}\n".format(self.list_mean_collision))
        fic.write("mean_death: {}\n".format(self.list_mean_death))
        fic.close()

        plt.figure("Mean of food collected")
        plt.plot(x_values, self.list_mean_food_collected, "b:x")
        plt.xlabel("Plays")
        plt.ylabel("Food")

        plt.figure("Mean of death by enemies")
        plt.plot(x_values, self.list_mean_death, "r:x")
        plt.xlabel("Number of training iteration done")
        plt.ylabel("Deaths")

        plt.figure("Mean of collisions")
        plt.plot(x_values, self.list_mean_collision, "g:x")
        plt.xlabel("Number of training iteration done")
        plt.ylabel("Collisions")

        plt.show()

    def test_iterations(self, verbose=False):
        """
        Run 50 iterations of the simulation in which the action of the agent is choosen by selecting the max Q-Value (Exploitation only)
        Compute mean collected foods / collisions / death

        Args:
            verbose (bool, optional): Print information or not. Defaults to False.
        """
        self.training_mode = False
        food_collected = 0
        self.nb_collision = 0
        self.nb_death = 0

        for test_cpt in range(self.number_test_env):
            self.initialize_grid()

            if (verbose and test_cpt == self.number_test_env - 1):
                self.verbose = True
                self.agent.verbose = True
            while (self.reset == False):
                self.run()

            food_collected += self.nb_collected_food

        food_collected /= self.number_test_env
        self.list_mean_food_collected.append(food_collected)
        print("self.list_mean_food_collected:", self.list_mean_food_collected)

        self.list_mean_collision.append(self.nb_collision / self.number_test_env)
        print("self.list_mean_collision", self.list_mean_collision)

        self.list_mean_death.append(self.nb_death / self.number_test_env)
        print("self.list_mean_death:", self.list_mean_death)

        self.training_mode = True