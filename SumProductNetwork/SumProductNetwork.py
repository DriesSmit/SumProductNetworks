# Die processing kan dalk baie vinniger gemaak word deur om direk na die variables te point

import numpy as np
import random
class SPN:
    def __init__(self):
        self.network = []
        self.varMap = {}
        self.variables = []

    def addVar(self, variables):
        for index, variable in enumerate(variables):
            self.variables.append(variable)
            self.varMap[variable.name] = index

    def addNodes(self,layer):
        self.network.append(layer)

    def increment_variables(self, want):
        adding = True
        index = len(want)-1
        while(adding):
            cur_want = want[index]
            variable = self.variables[self.varMap[cur_want]]

            var_index = variable.get_true_value() + 1
            if var_index < variable.get_size():
                variable.set_true_value(var_index)
                adding = False
            else:
                variable.set_true_value(0)
                index -= 1

    def set_all_var_values_one(self):
        for var_index, variable in enumerate(self.variables):
            for index in range(variable.get_size()):
                variable.set_value(index, 1.0)

    def gradient_decent(self, data, num_epochs=10, alpha = 0.1):
        # Also include batch size later on

        for epoch in range(num_epochs):
            for data_index in range(len(data)):

                # Forward pass

                # Set the input values
                for cur_var in data[data_index]:
                    variable = self.variables[self.varMap[cur_var[0]]]
                    variable.set_value(cur_var[1], 1.0)
                # Run the network
                answer = self.run_network()

                # Backwards pass
                first_grad = 1/answer # Moet daar nie teen deel deur nul beskerm word nie?
                self.network[len(self.network) - 1][0].pass_gradient(first_grad)
                #print("Network sum weights: ", self.network[len(self.network) - 1][0].weight_sum)
                #print("Network input x1: ", self.variables[0].get_value_point())
                #print("Network weights: ", self.network[len(self.network) - 1][0].probs)

            # Update the weights for this epoch
            for net_depth in range(len(self.network)):
                if type(self.network[net_depth][0]) == opp_sum:
                    for node_index in range(len(self.network[net_depth])):
                        self.network[net_depth][node_index].update_weights(alpha=alpha, num_data=len(data))

    def fit(self,data, method = "GD", num_epochs = 10, alpha = 0.1): # Method: GD (Gradient Decent), SEM (Soft Expectation Maximization) or HEM (Hard Expectation Maximization)

        self.init_probs()

        if method == "GD":
            self.gradient_decent(data, num_epochs=num_epochs, alpha = alpha)

        print("Final weights: ", self.network[len(self.network) - 1][0].probs)

    def run_network(self):
        # Run every node in the network
        for net_depth in range(len(self.network)):
            for node_index in range(len(self.network[net_depth])):
                self.network[net_depth][node_index].calc_value()
        return self.network[len(self.network) - 1][0].get_value_point()[0]

    # In this function any probability distribution can be calculated.
    def eval(self,want=[], given = []):
        # Apply Bayes' rule
        # Calculate the denominator
        den_value = 1.0
        self.set_all_var_values_one()
        if len(given) > 0:
            for cur_given in given:
                variable = self.variables[self.varMap[cur_given[0]]]
                for index in range(variable.get_size()):
                    if index != cur_given[1]:
                        variable.set_value(index, 0.0)
            den_value = self.run_network()
        # Calculate the numerator

        # Init all the wanted values and calculate the number of rows in the table
        num_cols = len(want) + 1# Add one for the probability value
        num_rows = 1
        for cur_want in want:
            variable = self.variables[self.varMap[cur_want]]
            num_rows *= variable.get_size()
            for index in range(variable.get_size()):
                variable.set_value(index, 0.0)
            variable.set_true_value(0)


        prob_table = np.zeros((num_rows, num_cols))

        # Populate the probability table
        for row_index in range(num_rows):
            for col_index, cur_want in enumerate(want):
                variable = self.variables[self.varMap[cur_want]]
                prob_table[row_index][col_index] = variable.get_true_value()
            prob_table[row_index][num_cols-1] = self.run_network()/den_value
            self.increment_variables(want)
        return prob_table

    def init_probs(self):
        for net_depth in range(len(self.network)):
            if type(self.network[net_depth][0]) == opp_sum:
                for node_index in range(len(self.network[net_depth])):
                    self.network[net_depth][node_index].init_probs()

class variable():
    def __init__(self,name="", size = 2):
        if size < 2:
            print("Size of ", size, " is invalid. Setting size to 2.")
            size = 2
        self.value = np.zeros(size)
        self.size = size
        self.name = name

        self.value_index = 0

    def set_value(self,index, value):
        self.value[index] = value

        if value >0.01 and self.value_index != index:
            self.value[self.value_index] = 0.0
            self.value_index = index

    # This value will be wrong if there is more that one non zero value in the variable.
    def set_true_value(self, value):
        self.value[self.value_index] = 0
        self.value[value] = 1
        self.value_index = value

    def get_value(self, index):
        return self.value[index]

    # This value will be wrong if there is more that one non zero value in the variable.
    def get_true_value(self):
        return self.value_index

    def get_value_point(self):
        return self.value

    def get_name(self):
        return self.name

    def get_size(self):
        return self.size

    def pass_gradient(self, value):
        pass

class opp_sum():
    def __init__(self, inherit=[], probs=[]):
        if len(inherit) != len(probs):
            raise ValueError("Dimensions not the same for inherit",len(inherit)," and probs: ", len(probs))
        self.inherit = []
        self.total = np.zeros(1)
        self.weight_sum = np.zeros(len(probs))

        for parent in inherit:
            if type(parent) == opp_multi:
                # This is a node [pointer, index]
                self.inherit.append((parent,0))
            elif len(parent) == 2:
                # This is a variable [pointer, index]
                self.inherit.append((parent[0], parent[1]))
            else:
                raise ValueError('To many input parameters from one parent. Found: ',len(parent))
        self.probs = probs

    def init_probs(self):
        sum_probs = 0.0
        for index in range(len(self.probs)):
            self.probs[index] = 0.5
            #random.random()
            sum_probs += self.probs[index]
        for index in range(len(self.probs)):
            self.probs[index] /= sum_probs

    def calc_value(self):
        self.total[0] = 0.0
        for index in range(len(self.inherit)):
            self.total[0] += self.inherit[index][0].get_value_point()[self.inherit[index][1]] * self.probs[index]

    def get_value_point(self):
        return self.total

    def pass_gradient(self, value):
        #print("At sum node: ", value, " ", self.total[0])
        for index in range(len(self.inherit)):
            self.weight_sum[index] += value * self.inherit[index][0].get_value_point()[self.inherit[index][1]]
            self.inherit[index][0].pass_gradient(value * self.probs[index])

    def update_weights(self, num_data=10, alpha=0.1):
        sum_of_probs = 0.0
        print("Probabilities before: ", self.probs)
        print("Sum weights: ", self.weight_sum)
        # Update the weights of the sum node

        gradients = self.weight_sum / num_data

        update_values = (gradients) - np.mean(gradients)

        for index in range(len(self.probs)):
            self.probs[index] += alpha * update_values[index] # Want to reach maximum likelihood

            if self.probs[index] <= 0.0:
                self.probs[index] = 0.0

            sum_of_probs += self.probs[index]
            self.weight_sum[index] = 0.0


        # Normalise the weights of the sum node
        for index in range(len(self.probs)):
            self.probs[index] /= sum_of_probs

        print("Probabilities after: ", self.probs)

class opp_multi():
    def __init__(self,inherit = []):
        self.inherit = []
        self.total = np.zeros(1)

        for parent in inherit:
            if type(parent) == opp_sum:
                # This is a node [pointer, index]
                self.inherit.append((parent, 0))
            elif len(parent) == 2:
                # This is a variable [pointer, index]
                self.inherit.append((parent[0], parent[1]))
            else:
                raise ValueError('To many input parameters from one parent. Found: ', len(parent))

    def calc_value(self):
        self.total[0] = 1.0
        for index in range(len(self.inherit)):
            self.total[0] *= self.inherit[index][0].get_value_point()[self.inherit[index][1]]

    def get_value_point(self):
        return self.total

    def pass_gradient(self, value):
        #print("At multiply node: ", value, " ", self.total[0])
        for index in range(len(self.inherit)):
            pre_node_grad = value * self.total[0] / (self.inherit[index][0].get_value_point()[self.inherit[index][1]] + 0.0000001)  # This is added to negate a division by zero
            self.inherit[index][0].pass_gradient(pre_node_grad)
