import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

m = 500

# single feature
size = np.random.randint(500, 3000, m).astype(float)

# true values (hidden from model)
# notice: non-linear relationship! price depends on size squared too
true_w1 = 0.05    # coefficient for size
true_w2 = 0.00003 # coefficient for size^2
true_b  = 50

# noise
noise = np.random.randn(m) * 10

# output — genuinely non-linear, linear regression alone won't fit this well
y = true_w1*size + true_w2*(size**2) + true_b + noise

# x is just one column for now
x = np.column_stack((size, size**2))
x = x.astype(float)

# initialization of x and y is done
def find_cost(w,b):
    error = (x @ w + b) - y
    return np.sum((error**2)/(2*m))

def find_dj_dw(error):
    dj_dw = np.array([0.0,0.0])
    dj_dw = (x.T @ error) / m
    return dj_dw

w = np.array([0.0,0.0])
b = 0

x[:,0] = (x[:,0]-np.mean(x[:,0]))/np.std(x[:,0])
x[:,1] = (x[:,1]-np.mean(x[:,1]))/np.std(x[:,1])
cost_history = []

i = 0
lr = 0.01
diff = 1

while(diff > 1e-8 and i < 50000):
    error = (x @ w + b) - y
    prev_cost = find_cost(w,b)
    dj_dw = find_dj_dw(error)

    dj_db = np.sum(error/m)

    temp_w = w - lr*dj_dw
    
    temp_b = b - lr*dj_db

    w = temp_w
    b = temp_b

    new_cost = find_cost(w,b)

    diff = abs(new_cost - prev_cost)
    cost_history.append(new_cost)

    i += 1

print(w[0])
print(w[1])
print(new_cost)
print(i)


    
    
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Decreasing")

plt.show()


#output with absolute max scaling
'''
0.054237273765470155
2.8886107706257674e-05
59.36591014746309
44205
x[:,0] = x[:,0]/1e3
x[:,1] = x[:,1]/1e6

0.06180475286347051
2.6784863958327666e-05
59.81476693324966
50000
x[:,0] = x[:,0]/3e3
x[:,1] = x[:,1]/9e6
'''

#============================================

#Z-score scaling
'''
37.39979974069448
71.05930453103967
59.365886269279514
35309

No rescaling done
'''