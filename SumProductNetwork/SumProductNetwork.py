# Die processing kan dalk baie vinniger gemaak word deur om direk na die variables te point

import numpy as np
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

    def get_prob(self):
        # Run every node in the network
        for net_depth in range(len(self.network)):
            for node_index in range(len(self.network[net_depth])):
                self.network[net_depth][node_index].calc_value()
        return self.network[len(self.network) - 1][0].get_value_point()[0]

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
            den_value = self.get_prob()
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
            prob_table[row_index][num_cols-1] = self.get_prob()/den_value
            self.increment_variables(want)
        return prob_table

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

        if value >0.01:
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

class opp_sum():
    def __init__(self, inherit = [], probs = []):
        if len(inherit) != len(probs):
            raise ValueError("Deminsions not the same for inherit",len(inherit)," and probs: ", len(probs))
        self.inherit = []
        self.total = np.zeros(1)

        for parent in inherit:
            if type(parent) == opp_multi:
                # This is a node [pointer, index]
                self.inherit.append((parent.get_value_point(),0))
            elif len(parent) == 2:
                # This is a variable [pointer, index]
                self.inherit.append((parent[0].get_value_point(), parent[1]))
            else:
                raise ValueError('To many input parameters from one parent. Found: ',len(parent))
        self.probs = probs

    def calc_value(self):
        self.total[0] = 0.0
        for index in range(len(self.inherit)):
            self.total[0] += self.inherit[index][0][self.inherit[index][1]] * self.probs[index]

    def get_value_point(self):
        return self.total

class opp_multi():
    def __init__(self,inherit = []):
        self.inherit = []
        self.total = np.zeros(1)

        for parent in inherit:
            if type(parent) == opp_sum:
                # This is a node [pointer, index]
                self.inherit.append((parent.get_value_point(), 0))
            elif len(parent) == 2:
                # This is a variable [pointer, index]
                self.inherit.append((parent[0].get_value_point(), parent[1]))
            else:
                raise ValueError('To many input parameters from one parent. Found: ', len(parent))

    def calc_value(self):
        self.total[0] = 1.0
        for index in range(len(self.inherit)):
            self.total[0] *= self.inherit[index][0][self.inherit[index][1]]

    def get_value_point(self):
        return self.total
