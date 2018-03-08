from SumProductNetwork import SPN
from SumProductNetwork import opp_sum, opp_multi, variable

spn = SPN()

network = []

# Initialize variables
n_x1 = variable()
x1 = variable()
n_x2 = variable()
x2 = variable()

# The plus operators
p1 = opp_sum(name = "p1",inherit=[n_x1,x1],probs=[0.2, 0.8])
p2 = opp_sum(name = "p2",inherit=[n_x1,x1],probs=[0.1, 0.9])
p3 = opp_sum(name = "p3",inherit=[n_x2,x2],probs=[0.4, 0.6])
network.append((p1, p2, p3))

# The multiply operators
m1 = opp_multi(name = "m1",inherit=[p1,p3])
m2 = opp_multi(name = "m2",inherit=[p2,p3])
network.append((m1, m2))

# The final plus operator
p4 = opp_sum(name = "p1",inherit=[m1,m2],probs=[0.3, 0.7])
network.append([p4])

spn.setStructure(network=network, variables = [n_x1, x1, n_x2, x2])

variables = [1.0,0.0,0.0,1.0]

# -------------------
# |n_x1 x1 n_x2 x2|   p   |
# | 0   0   0   0 | 0.000 |
# | 0   0   0   1 | 0.000 |
# | 0   0   1   0 | 0.000 |
# | 0   0   1   1 | 0.000 |
# | 0   1   0   0 | 0.000 |
# | 0   1   0   1 | 0.522 |
# | 0   1   1   0 | 0.348 |
# | 0   1   1   1 | 0.870 |
# | 1   0   0   0 | 0.000 |
# | 1   0   0   1 | 0.078 |
# | 1   0   1   0 | 0.052 |
# | 1   0   1   1 | 0.130 |
# | 1   1   0   0 | 0.000 |
# | 1   1   0   1 | 0.600 |
# | 1   1   1   0 | 0.400 |
# | 1   1   1   1 | 1.000 |

print("The probability is: ", spn.eval(variables))
