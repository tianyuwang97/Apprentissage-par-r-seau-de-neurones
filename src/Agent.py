import copy
import math
import random
from Neural_network import Neural_network

class Agent:

    def __init__(self):
        self.id = 0
        self.symbol = "I"
        self.action_list = ["North", "South", "East", "West"]
        self.position = None
        self.old_position = None
        self.energy_max = 160
        self.energy_init = 40
        self.energy = self.energy_init

        self.sensors_id = ["Y", "O", "X", "o"]

        # Number of each sensor type
        self.nb_X = 12
        self.nb_O = 20
        self.nb_Y = 20
        self.nb_o = 40

        # Initialize the sensors' input
        self.init_sensor_value()

        # Set sensors' position around the agent
        self.X_position = self.merge_layer_sensor(self.place_layer_sensor(2, 1), self.place_layer_sensor(1, 1))
        self.O_position = self.merge_layer_sensor(self.place_layer_sensor(6, 2), self.place_layer_sensor(4, 2))
        self.Y_position = self.place_layer_sensor(10, 2)
        self.o_position = []
        for i in range(4, 0, -1):
            self.o_position = self.merge_layer_sensor(self.o_position, self.place_layer_sensor(i, 1))

        # Set sensors' scope
        self.X_scope = self.place_layer_sensor(1, 1)
        self.X_scope.insert(0, (0, 0))
        self.O_scope = []
        for x in range(-1, 2, 1):
            for y in range(-1, 2, 1):
                self.O_scope.append((x,y))
        self.Y_scope = self.merge_layer_sensor(self.place_layer_sensor(2, 1), self.place_layer_sensor(1, 1))
        self.Y_scope.insert(0, (0, 0))
        self.o_scope = [(0, 0)]

        # Energy level is coarse coded using 16 input units. Each of them represents an energy level and is activated when the agent's energy is close to it (page 15)
        self.input_energy_value = None

        # Define the value of collide (collision to obstacles) for input
        self.input_collide_value = 0

        # Last action, 0 or 1, for North, South, East, West
        self.last_action = [0, 0, 0, 0]

        # Input and value of the action chosen
        self.input_vector_chosen = None
        self.value_output = None
        self.discount_factor = 0.9 

        self.utility_network = Neural_network()
        self.temperature = 1 / 20

        self.verbose = False

    def init_sensor_value(self):
        """
        Init all sensors' values with 0
        """
        self.food_X_values = []
        self.food_O_values = []
        self.food_Y_values = []
        self.enemy_O_values = []
        self.enemy_X_values = []
        self.obstacle_o_values = []

        for i in range(self.nb_X):
            self.food_X_values.append(0)
            self.enemy_X_values.append(0)
        for i in range(self.nb_O):
            self.food_O_values.append(0)
            self.enemy_O_values.append(0)
        self.food_Y_values = [0 for i in range(self.nb_Y)]
        self.obstacle_o_values = [0 for i in range(self.nb_o)]

    def place_layer_sensor(self, distance, step):
        """
        Place sensors in a square shape as seen in figure 8, page 14

        Args:
            distance (int): manhattan distance of the sensors to the agent
            step (int): space between each sensors

        Returns:
            (list<tuple<int>>): list of every sensor's position
        """
        organized_list = []
        x_pos = -distance
        y_pos = 0

        # Top-left side
        while x_pos < 0:
            organized_list.append((x_pos, y_pos))
            x_pos += step
            y_pos += step

        # Top-right side
        while y_pos > 0:
            organized_list.append((x_pos, y_pos))
            x_pos += step
            y_pos -= step

        # Bottom-right side
        while x_pos > 0:
            organized_list.append((x_pos, y_pos))
            x_pos -= step
            y_pos -= step

        # Bottom-left side
        while y_pos < 0:
            organized_list.append((x_pos, y_pos))
            x_pos -= step
            y_pos += step

        return organized_list

    def merge_layer_sensor(self, first_list, second_list):
        """
        Function used to merge list of sensors' position their positions represent a multi-layered square (X, O and o sensors)

        Args:
            first_list (list<tuple<int>>): list of sensors' position
            second_list (list<tuple<int>>): list of sensors' position
        Returns:
            list<tuple<int>> : list of every sensor's position
        """
        l1_quarter = int(len(first_list) / 4)
        l2_quarter = int(len(second_list) / 4)

        merged_list = []
        for i in range(4):
            merged_list += first_list[i * l1_quarter: (i + 1) * l1_quarter]
            merged_list += second_list[i * l2_quarter: (i + 1) * l2_quarter]

        return merged_list

    def rotate_state_inputs(self, list_values, orientation):
        """
        Rotating the input as explained page 16 for the global representation

        Args:
            list_values (list<integer>) : List of values
            orientation (String): Orientation for the rotation

        Returns:
            (List): rotated list
        """
        quarter = int(len(list_values) / 4)
        quarter_nb = 0
        if orientation == "East":
            quarter_nb = 1
        elif orientation == "South":
            quarter_nb = 2
        elif orientation == "West":
            quarter_nb = 3
        return copy.deepcopy(list_values[quarter_nb * quarter:]) + copy.deepcopy(list_values[:quarter_nb * quarter])

    def create_input_vector(self,orientation):
        """
        Rotating the input as explained page 16 for the global representation

        Args:
            orientation (String): Orientation for the rotation

        Returns:
            (List): Input vector for the neural network
        """
        input_vector = []
        input_vector += self.rotate_state_inputs(self.enemy_O_values,orientation)
        input_vector += self.rotate_state_inputs(self.enemy_X_values, orientation)
        input_vector += self.rotate_state_inputs(self.food_Y_values, orientation)
        input_vector += self.rotate_state_inputs(self.food_O_values, orientation)
        input_vector += self.rotate_state_inputs(self.food_X_values, orientation)
        input_vector += self.rotate_state_inputs(self.obstacle_o_values, orientation)
        input_vector += self.input_energy_value
        input_vector += self.rotate_state_inputs(self.last_action, orientation)
        input_vector += [self.input_collide_value]
        return input_vector

    def choose_action(self, training_mode):
        """
        Compute the next action

        Args:
            training_mode ([type]): [description]

        Returns:
            [String]: Action choosen ("North", "South", "East", "West")
        """
        input_dict, output_dict = self.propagation()
        if self.verbose :
            print("dico output", output_dict)

        if training_mode:
            action = self.stochastic_action_selector(output_dict)
        else:
            action = max(output_dict, key=output_dict.get)

        self.input_vector_chosen = input_dict[action]
        self.value_output = output_dict[action]

        if self.verbose:
            print("AGENT - Action taken : ", action)

        return action

    def propagation(self):
        """
        Give the input vector to the neural network and compute the output

        Returns:
            [List<float>]: Inputs of the network
            [List<float>]: Outputs of the network
        """
        input_dict = {}
        output_dict = {}
        for orientation in self.action_list:
            input_dict[orientation] = self.create_input_vector(orientation)
            output_dict[orientation] = self.utility_network.forward_propagation(input_dict[orientation])
        return input_dict, output_dict

    def stochastic_action_selector(self, output_dict,training_mode = None):
        """
        Compute probabilites for each action as defined in equation 4 page 5.
        Select an action with those probabilites
        Args:
            output_dict (List<float>): Outputs of the neural network
            training_mode (Bool): True if training, False if testing. Defaults to None.

        Returns:
            [String]: Action choosen ("North", "South", "East", "West")
        """
        prob_action_list = {}
        for action in self.action_list:
            prob_action_list[action] = math.exp(output_dict[action] / self.temperature)
        somme = sum(prob_action_list.values())
        for action in self.action_list:
            prob_action_list[action] /= somme
        
        if self.verbose :
            print("proba_action stochastic : ",prob_action_list)
        
        sorted_prob_action_list = []
        for _, value in prob_action_list.items():
            sorted_prob_action_list.append(value)
        sorted_prob_action_list = sorted(sorted_prob_action_list)

        # Select the action
        draw = random.random()
        proba_sum = 0

        for i in range(len(sorted_prob_action_list)):
            proba_sum += sorted_prob_action_list[i]
            if draw < proba_sum:
                for key, value in prob_action_list.items():
                    if value == sorted_prob_action_list[i]:
                        return self.action_list[self.action_list.index(key)]

    def backpropagate_reward_error(self, reward: float):
        """
        Compute reward error and backpropagate it.
        Refers line 4 and 5 of figure 4, page 6

        Args:
            reward (float): reward from last action
        """

        _, output_dict = self.propagation()
        new_reward = reward + self.discount_factor * max(output_dict.values())
        if self.verbose :
            print("Retropropagation with the reward : ",new_reward)
        self.utility_network.backpropagation(self.input_vector_chosen, new_reward)

    def get_id(self):
        return self.id

    def get_symbol(self):
        return self.symbol

    def get_action_list(self):
        return self.action_list

    def set_position(self, new_position: tuple):
        self.position = new_position

    def get_position(self):
        return self.position

    def set_old_position(self, new_old_position: tuple):
        self.old_position = new_old_position

    def get_energy(self):
        return self.energy

    def set_energy(self, val):
        self.energy = min(val, self.energy_max)

    def add_energy(self):
        self.energy = min(self.energy + 15, self.energy_max)

    def consume_energy(self):
        self.energy -= 1

    def get_energy_init(self):
        return self.energy_init

    def get_max_energy(self):
        return self.energy_max

    # Set sensors' value
    def set_food_X_values(self, new_values):
        self.food_X_values = new_values

    def set_food_Y_values(self, new_values):
        self.food_Y_values = new_values

    def set_food_O_values(self, new_values):
        self.food_O_values = new_values

    def set_enemy_O_values(self, new_values):
        self.enemy_O_values = new_values

    def set_enemy_X_values(self, new_values):
        self.enemy_X_values = new_values

    def set_obstacle_o_values(self, new_values):
        self.obstacle_o_values = new_values

    # Get sensors' value
    def get_food_X_values(self):
        return self.food_X_values

    def get_food_Y_values(self):
        return self.food_Y_values

    def get_food_O_values(self):
        return self.food_O_values

    def get_enemy_O_values(self):
        return self.enemy_O_values

    def get_enemy_X_values(self):
        return self.enemy_X_values

    def get_obstacle_o_values(self):
        return self.obstacle_o_values

    # Get sensors' position
    def get_X_position(self):
        return self.X_position

    def get_O_position(self):
        return self.O_position

    def get_Y_position(self):
        return self.Y_position

    def get_o_position(self):
        return self.o_position

    # Get sensors' scope
    def get_X_scope(self):
        return self.X_scope

    def get_O_scope(self):
        return self.O_scope

    def get_Y_scope(self):
        return self.Y_scope

    def get_o_scope(self):
        return self.o_scope

    def get_sensors_id(self):
        return self.sensors_id

    def set_last_action(self, new_action):
        self.last_action = new_action

    def set_input_energy(self, new_input_energy):
        self.input_energy_value = new_input_energy

    def set_collide(self, new_collide_value):
        self.input_collide_value = new_collide_value