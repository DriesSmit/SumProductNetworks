from SumProductNetwork import SPN
from SumProductNetwork import opp_sum, opp_multi, variable
import numpy as np
total_score = 0
total = 0

# Test inference on the network

spn = SPN()

# Initialize variables
x1 = variable(name='x1', size=2)
x2 = variable(name='x2', size=2)
spn.addVar((x1, x2))

# The plus operators
p1 = opp_sum(inherit=[(x1, 0), (x1, 1)], probs=[0.2, 0.8])
p2 = opp_sum(inherit=[(x1, 0), (x1, 1)], probs=[0.1, 0.9])
p3 = opp_sum(inherit=[(x2, 0), (x2, 1)], probs=[0.4, 0.6])
spn.addNodes((p1, p2, p3))

# The multiply operators
m1 = opp_multi(inherit=[p1, p3])
m2 = opp_multi(inherit=[p2, p3])
spn.addNodes((m1, m2))

# The final plus operator
p4 = opp_sum(inherit=[m1, m2], probs=[0.3, 0.7])
spn.addNodes([p4])

test_probs = []

want = ['x2']
given = []
ans = np.array([[0.0, 0.4], [1.0, 0.6]])
test_probs.append((want,given,ans))

want = ['x1']
given = [('x2',1)]
ans = np.array([[ 0.0,0.13],[ 1.0,0.87]])
test_probs.append((want,given,ans))

want = ['x1', 'x2']
given = []
ans = np.array([[0.0,0.0,0.052],[0.0,1.0,0.078],[1.0,0.0,0.348],[1.0,1.0,0.522]])
test_probs.append((want,given,ans))

for index in range(len(test_probs)):
    want, given, answer = test_probs[index]
    infer_val = spn.eval(want=want, given = given)
    if (infer_val.round(10)==answer).all():
        total_score += 1
        total += 1
    else:
        print("Inference check. Values do not match at index: ", index)
        print("True values:\n\n", answer, "\n\nCalc values:\n\n", infer_val)
        total += 1

print("Inference check completed : ", total_score, "/", total)


# Test learning on the network

print("\nTotal number of tests passed: ", total_score, "/", total)

if total == total_score:
    print("All tests were passed.")
else:
    print("Failure in some tests detected.")