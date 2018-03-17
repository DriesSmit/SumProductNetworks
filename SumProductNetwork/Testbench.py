from SumProductNetwork import SPN
from SumProductNetwork import opp_sum, opp_multi, variable
import numpy as np
spn = SPN()

total_score = 0
total = 0

# Test the inference on the network

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
p4 = opp_sum(name ="p1", inherit=[m1,m2],probs=[0.3, 0.7])
network.append([p4])

spn.setStructure(network=network, variables = [n_x1, x1, n_x2, x2])

# |---------------|-------|
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
# |-----------------------|

full_prob_dist = np.zeros((16,5))

full_prob_dist[0] = [0, 0, 0, 0, 0.000]
full_prob_dist[1] = [0, 0, 0, 1, 0.000]
full_prob_dist[2] = [0, 0, 1, 0, 0.000]
full_prob_dist[3] = [0, 0, 1, 1, 0.000]
full_prob_dist[4] = [0, 1, 0, 0, 0.000]
full_prob_dist[5] = [0, 1, 0, 1, 0.522]
full_prob_dist[6] = [0, 1, 1, 0, 0.348]
full_prob_dist[7] = [0, 1, 1, 1, 0.870]
full_prob_dist[8] = [1, 0, 0, 0, 0.000]
full_prob_dist[9] = [1.0, 0.0, 0.0, 1.0, 0.078]
full_prob_dist[10] = [1, 0, 1, 0, 0.052]
full_prob_dist[11] = [1, 0, 1, 1, 0.130]
full_prob_dist[12] = [1, 1, 0, 0, 0.000]
full_prob_dist[13] = [1, 1, 0, 1, 0.600]
full_prob_dist[14] = [1, 1, 1, 0, 0.400]
full_prob_dist[15] = [1, 1, 1, 1, 1.000]

for index in range(len(full_prob_dist)):
    variables = full_prob_dist[index][:4]
    infer_val = spn.eval(variables)
    if round(infer_val,16) == full_prob_dist[index][4]:
        total_score += 1
        total += 1
    else:
        print("Inference check. Values do not match at index: ", index)
        print("True value: ", full_prob_dist[index][4], ". Calc value: ", infer_val)
        total += 1

print("Inference check completed : ", total_score, "/", total)

print("\nTotal number of tests passed: ", total_score, "/", total)

if total == total_score:
    print("All tests were passed.")
else:
    print("Failure in some tests detected.")