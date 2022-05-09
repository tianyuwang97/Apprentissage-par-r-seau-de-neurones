import random
import math
import copy


class Neural_network:
    def __init__(self):
        """
        As defined in pages 14 and 19, the neural network is a three layered one with 145 input, 30 hidden and 1 output
        Learning rate and momentum are defined in pages 18-19
        """
        self.nb_layers = 3
        self.nb_neurons_per_layers = [145, 30, 1]
        self.learning_rate = 0.3
        self.momentum = 0.9

        self.weights = [] # 2d array, weights of connections from the previous layer (length = nb_layer - 1 since the input layer has no previous layer)
        self.values = [] # 2d array, value of every neurones of every layer (length = nb_layer)
        self.weighted_sums = [] # 2d array sums of values received by the previous layer (length = nb_layer - 1 since there is no weighted sums for the first layer)
        self.weighted_variations = [] # 2d array use for the momentum: delta_w_ij(n-1) = learning_rate * delta_j(n-1) * y_i(n-1)

        # Init input layer
        self.values.append([0 for i in range(self.nb_neurons_per_layers[0])])

        # Init hidden and output layers
        for i in range(self.nb_layers - 1):
            self.weights.append([])
            self.values.append([])
            self.weighted_sums.append([])
            self.weighted_variations.append([])
            for j in range(self.nb_neurons_per_layers[i + 1]):
                self.weights[i].append([random.uniform(-0.05, 0.05) for k in range(self.nb_neurons_per_layers[i])])
                self.values[i + 1].append(0)
                self.weighted_sums[i].append(0)
                self.weighted_variations[i].append([0 for k in range(self.nb_neurons_per_layers[i])])

    def forward_propagation(self, inputs):
        """
        Propagate values from the inputs to the outputs of the network

        Args:
            inputs (List<Integer>): Input vector (list of 0 or 1 whether the input neurone is activated or not)

        Returns:
            [List<float>]: Output of the network (Q-Values of each action)
        """
        self.values[0] = inputs

        for i in range(self.nb_layers - 1): # for each layer (except the input layer)
            for j in range(len(self.weights[i])): # For each neurone
                self.weighted_sums[i][j] = self.compute_weight_sum(self.weights[i][j], self.values[i]) # compute the weighted sums
                self.values[i + 1][j] = self.squashing_function(self.weighted_sums[i][j]) # compute value of the neurone

        return copy.deepcopy(self.values[-1][0])

    def backpropagation(self, input_vector_choosen, new_reward):
        self.forward_propagation(input_vector_choosen)

        # Used by the backpropagation to update the weigths
        self.tmp_weights = copy.deepcopy(self.weights)

        new_reward = self.truncate(new_reward)
        new_reward = [new_reward]

        # Backpropagation on the output layer
        gradients = self.output_layer_backpropagation(self.weighted_sums[-1], self.weights[-1],
                                                        self.values[-1], self.values[self.nb_layers - 2],
                                                        new_reward,self.learning_rate, self.weighted_variations[-1],
                                                        self.momentum)

        # Backpropagation on the hidden layer
        for i in reversed(range(self.nb_layers - 2)):
            # print("Layer :", i)
            gradients = self.hidden_layer_backpropagation(self.weighted_sums[i], self.weights[i],
                                                            self.weights[i + 1], self.values[i], gradients,
                                                            self.learning_rate, self.weighted_variations[i],
                                                            self.momentum, i)

        # Update of the weights
        self.weights = copy.deepcopy(self.tmp_weights)

    def output_layer_backpropagation(self, layer_weighted_sums, layer_weights,layer_outputs, layer_inputs, layer_targets,
                                        learning_rate, layer_weights_variations, momentum):
        gradients_list = []
        for i in range(len(layer_weights)):
            error = self.compute_output_layer_error(layer_outputs[i], layer_targets[i])
            gradients_list.append(self.gradient_calculation(error, layer_weighted_sums[i]))
            for j in range(len(layer_weights[i])):
                self.tmp_weights[-1][i][j] = layer_weights[i][j] \
                                          + learning_rate * gradients_list[i] * layer_inputs[j] \
                                          + momentum * layer_weights_variations[i][j]
                layer_weights_variations[i][j] = learning_rate * gradients_list[i] * layer_inputs[j]

        return gradients_list

    def hidden_layer_backpropagation(self, layer_weighted_sums, layer_weights,
                                        next_layer_weights, layer_inputs, gradients,
                                        learning_rate, layer_weights_variations, momentum, k):
        # print("Hidden layer retropropagation ")
        gradients_list = []
        weight_i_j = []
        for i in range(len(layer_weights)):
            # Weights of the i neuron to the next layer
            for j in range(len(gradients)):
                weight_i_j.append(next_layer_weights[j][i])
            error = self.compute_hidden_layer_error(gradients, weight_i_j)
            weight_i_j = []
            gradients_list.append(self.gradient_calculation(error, layer_weighted_sums[i]))
            for j in range(len(layer_weights[i])):
                self.tmp_weights[k][i][j] = layer_weights[i][j] \
                                          + learning_rate * gradients_list[i] * layer_inputs[j] \
                                          + momentum * layer_weights_variations[i][j]
                layer_weights_variations[i][j] = learning_rate * gradients_list[i] * layer_inputs[j]
        return gradients_list

    def compute_weight_sum(self, weight_list, value_list):
        res = 0
        for i in range (len(weight_list)):
            res += weight_list[i] * value_list[i]
        return res

    def squashing_function(self,neurone_input_weighted_sum):
        """
        The squashing function defined page 13

        Args:
            weighted_sum_param ([type]): [description]
        """
        # Article's function multiplied by 2 for an output defined in  [-1.0 ; 1.0]
        return ((1.0 / (1.0 + math.exp(-neurone_input_weighted_sum))) - 0.5) * 2.0

    def squashing_function_derivative(self, weighted_sum_param):
        return 2 * math.exp(-weighted_sum_param) / ((1 + math.exp(-weighted_sum_param))**2)

    def compute_output_layer_error(self, output, target):
        return target - output

    def gradient_calculation(self, error, weighted_sum):
        return error * self.squashing_function_derivative(weighted_sum)

    def compute_hidden_layer_error(self, gradients, weigths_list):
        return self.compute_weight_sum(weigths_list, gradients)

    def truncate(self, value):
        if value > 1.0:
            return 1.0
        elif value < -1.0:
            return -1.0
        else:
            return value
