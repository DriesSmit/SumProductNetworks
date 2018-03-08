# Die processing kan dalk baie vinniger gemaak word deur om direk na die variables te point

import numpy as np
class SPN:
    def __init__(self):
        pass

    def setStructure(self, network, variables):
        self.input_values = variables
        self.network = network
    # Each variable can be in one of its states or 'U' for unknown
    def eval(self,values):
        # Initialize the input values
        for index in range(len(self.input_values)):
            self.input_values[index].set_value(values[index])

        # Run every node in the network
        for net_depth in range(len(self.network)):
            for node_index in range(len(self.network[net_depth])):
                self.network[net_depth][node_index].calc_value()
        return self.network[len(self.network)-1][0].get_value()

    # def fit(self, data):
    #     pass
    def get(self, data):
        pass
    #
    # def findStructure(self, data):
    #     pass
    #
    # def loadNetwork(self, filename = ""):
    #     pass
    #
    # def saveNetwork(self, filename = ""):
    #     pass

class variable():
    def set_value(self,value):
        self.value = value
    def get_value(self):
        return self.value

class opp_sum():
    def __init__(self,name="", inherit = [], probs = []):
        if len(inherit) != len(probs):
            print("Deminsions not the same for inherit",len(inherit)," and probs: ", len(probs))
            return None
        self.name = name
        self.inherit = inherit
        self.probs = probs

    def calc_value(self):
        self.total = 0.0
        for index in range(len(self.inherit)):
            self.total += self.inherit[index].get_value() * self.probs[index]
        return self.total

    def get_name(self):
        return self.name

    def get_value(self):
        return self.total

class opp_multi():
    def __init__(self,name="", inherit = []):
        self.name = name
        self.inherit = list(inherit)

    def calc_value(self):
        self.total = 1.0
        for index in range(len(self.inherit)):
            self.total *= self.inherit[index].get_value()
        return self.total

    def get_name(self):
        return self.name

    def get_value(self):
        return self.total
