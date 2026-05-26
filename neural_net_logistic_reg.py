import numpy as np
import math
import matplotlib.pyplot as plt

np.random.seed(42)

m = 500

# features
tumor_size = np.random.uniform(0.5, 10, m)      # size in cm
patient_age = np.random.uniform(20, 80, m)       # age in years

# XOR-like pattern — neural network needed, logistic regression will fail
# malignant if: (large tumor AND old patient) OR (small tumor AND young patient)
y = ((tumor_size > 5) & (patient_age > 50) |
     (tumor_size < 3) & (patient_age < 35)).astype(float)

# feature matrix
x = np.column_stack((tumor_size, patient_age))

# initialization done
# initialization done

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def cost_find(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)  # prevent log(0)
    cost = (-1/m)*(y@np.log(p) + (1-y)@np.log(1-p))
    return cost

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def find_p(a, w2, b2):
    p = a@w2 + b2
    return sigmoid(p)

def find_a(w1, b1):
    z1 = x @ w1.T + b1
    return relu(z1)

def find_dj_dw1(a, p):
    z1 = x @ w1.T + b1
    delta = np.outer(p - y, w2) * relu_derivative(z1)
    return (delta.T @ x) / m

def find_dj_db1(a, p):
    z1 = x @ w1.T + b1
    delta = np.outer(p - y, w2) * relu_derivative(z1)
    return np.sum(delta, axis=0) / m

def find_dj_dw2(a, p):
    return (a.T @ (p - y)) / m          # (3,500) @ (500,) = (3,)

def find_dj_db2(p):
    dj_db2 = (1/m)*(np.sum((p-y)))
    return dj_db2


x[:,0] = x[:,0]/10
x[:,1] = x[:,1]/80
y = y.astype(float)

w1 = np.random.randn(8, 2) * 0.1
w2 = np.random.randn(8) * 0.1
b1 = np.zeros(8)
b2 = 0.0

a = find_a(w1,b1)
p = find_p(a, w2, b2)

cost_history = []

i = 0
diff = 1
lr = 0.1

while(diff > 1e-8 and i < 100000):
    a = find_a(w1,b1)
    p = find_p(a, w2, b2)
    cost = cost_find(p)
    dj_dw1 = find_dj_dw1(a,p)
    dj_dw2 = find_dj_dw2(a,p)
    dj_db1 = find_dj_db1(a,p)
    dj_db2 = find_dj_db2(p)

    temp_w1 = w1 - lr*dj_dw1
    temp_w2 = w2 - lr*dj_dw2
    temp_b1 = b1 - lr*dj_db1
    temp_b2 = b2 - lr*dj_db2

    w1 = temp_w1
    w2 = temp_w2
    b1 = temp_b1
    b2 = temp_b2

    a = find_a(w1,b1)
    p = find_p(a, w2, b2)

    new_cost = cost_find(p)
    diff = abs(new_cost - cost)
    cost_history.append(new_cost)

    i += 1

print(i)
print(new_cost)


plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Decreasing")

plt.show()

# create a grid covering the feature space
ts_grid = np.linspace(0.5, 10, 200)
age_grid = np.linspace(20, 80, 200)
TS, AGE = np.meshgrid(ts_grid, age_grid)

# flatten and scale
grid = np.column_stack((TS.ravel()/10, AGE.ravel()/80))

# predict on grid
a_grid = relu(grid @ w1.T + b1)
p_grid = sigmoid(a_grid @ w2 + b2)
p_grid = p_grid.reshape(200, 200)

# plot
plt.figure()
plt.contourf(TS, AGE, p_grid, levels=[0, 0.5, 1], alpha=0.3, colors=['blue','red'])
plt.scatter(tumor_size[y==1], patient_age[y==1], color='red', label='Malignant', alpha=0.5)
plt.scatter(tumor_size[y==0], patient_age[y==0], color='blue', label='Benign', alpha=0.5)
plt.contour(TS, AGE, p_grid, levels=[0.5], colors='green')
plt.xlabel("Tumor Size")
plt.ylabel("Age")
plt.title("Neural Network Decision Boundary")
plt.legend()
plt.show()

# WORKING
'''
DATASET
→ tumor_size, patient_age generated randomly
→ y defined by XOR pattern (two separate malignant regions)
→ x = column stack of both features
→ feature scaling: divide by 10 and 80

FORWARD PASS
→ find_a: x @ w1.T + b1 → ReLU → hidden activations (500,8)
→ find_p: a @ w2 + b2 → sigmoid → final prediction (500,)

COST
→ log loss on predictions p vs true labels y
→ np.clip prevents log(0) crash

BACKPROPAGATION
→ dw2, db2: output layer gradients (same as logistic regression)
→ dw1, db1: hidden layer gradients using chain rule
           = output error × w2 × relu_derivative × input x

TRAINING LOOP
→ forward pass → cost → gradients → update weights → repeat
→ stop when cost stops changing (diff < 1e-8) or 100k iterations

DECISION BOUNDARY GRAPH
→ create 200×200 grid covering entire feature space
→ run forward pass on every grid point
→ color regions based on prediction
→ green line = where prediction = exactly 0.5

'''

#============================================


#output with absolute max scaling
'''
8 neurons
66771
0.026974572780628328

16 neurons
38442
0.24845179470896092
'''

#============================================

#output with min max scaling
'''
14424
0.04408758248039432
'''

#============================================


#Z-score scaling
'''
100000
0.004517118282524217

No rescaling done
'''
