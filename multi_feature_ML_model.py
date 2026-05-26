import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)

m = 500

# features
size = np.random.randint(500, 3000, m)
bedrooms = np.random.randint(1, 6, m)

# stack features into matrix (IMPORTANT)
x = np.column_stack((size, bedrooms))

# true values (hidden from model)
true_w1 = 0.1
true_w2 = 20
true_b = 0

# noise
noise = np.random.randn(m) * 10

# output
y = true_w1*size + true_w2*bedrooms + true_b + noise

# initialization of x and y is done

def find_cost(w,b):
    error = (x @ w + b) - y
    return np.sum((error**2)/(2*m))

def find_dj_dw(error):
    dj_dw = np.array([0.0,0.0])
    for j in (0,1):
        dj_dw[j] = np.sum(error*x[:,j])/m
    return dj_dw


w = np.array([0.0,0.0])
b = np.random.randint(0,10)

x = x.astype(float)
x0 = x[:,0]
x1 = x[:,1]
x[:,0] = (x[:,0]-np.mean(x[:,0]))/np.std(x[:,0])
x[:,1] = (x[:,1]-np.mean(x[:,1]))/np.std(x[:,1])

#y = y/350

cost_history = []

i = 0
diff = 1
lr = 0.05

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


print((w[0]))
print((w[1]))
print(b)
print(new_cost)
print(i)


    
    
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Decreasing")

plt.show()


#output with absolute max scaling
'''
0.10034399160691694
20.11265966129283
0.0352168596295145
47.748080960821696
5558
'''

#============================================

#output with min max scaling
'''
0.10034473873767757
20.112665157746502
70.31889286047529
47.74807997296563
3643
'''
#scaling and rescaling 
'''
x[:,0] = (x[:,0]-500)/2500
x[:,1] = (x[:,1]-1)/5

print((w[0]/2500))
print((w[1]/5))
'''

#============================================


#Z-score scaling
'''
69.0147591440117
29.131679136715707
237.43392342608797
47.74807825682313
259

No rescaling done
'''


