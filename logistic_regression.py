import numpy as np
import math
import matplotlib.pyplot as plt

np.random.seed(42)

m = 500

# features
tumor_size = np.random.uniform(0.5, 10, m)      # size in cm
patient_age = np.random.uniform(20, 80, m)       # age in years

# true weights (hidden from model)
true_w1 = 0.6    # tumor size has strong influence
true_w2 = 0.04   # age has weaker influence
true_b  = -4

# raw score (like linear regression output)
z = true_w1*tumor_size + true_w2*patient_age + true_b

# convert to probability using sigmoid
prob = 1 / (1 + np.exp(-z))

# final label — 1 = malignant, 0 = benign
y = (prob > 0.5).astype(int)
flip = np.random.rand(m) < 0.1
y = np.where(flip, 1 - y, y)

# feature matrix
x = np.column_stack((tumor_size, patient_age))

# initialization done

def cost_find(w,b):
    f_x = 1/(1+((math.e)**(-(x@w + b))))
    cost = (-1/m)*(y@np.log(f_x) + (1-y)@np.log(1-f_x))
    return cost

def find_dj_dw(w,b):
    f_x = 1/(1+((math.e)**(-(x@w + b))))
    error = f_x - y
    dj_dw = np.array([0.0,0.0])
    dj_dw = (x.T @ error) / m
    return dj_dw

x[:,0] = (x[:,0]/10)
x[:,1] = (x[:,1]/80)

y = y.astype(float)

w = np.array([0.0,0.0])
b = 0.0

f_x = 1/(1+((math.e)**(-(x@w + b))))
cost_history = []

i = 0
diff = 1
lr = 0.05

while(diff > 1e-8 and i < 100000):
    f_x = 1/(1+((math.e)**(-(x@w + b))))
    cost = cost_find(w,b)
    dj_dw = find_dj_dw(w,b)

    dj_db = np.sum((f_x - y)/m)

    temp_w = w - lr*dj_dw
    
    temp_b = b - lr*dj_db

    w = temp_w
    b = temp_b

    new_cost = cost_find(w,b)
    diff = abs(new_cost - cost)
    cost_history.append(new_cost)

    i += 1

w[0] = w[0]/10
w[1] = w[1]/80

x[:,0] = (x[:,0]*10)
x[:,1] = (x[:,1]*80)

print(w)
print(b)
print(new_cost)
print(i)
    
plt.figure()
plt.scatter(tumor_size[y==1], patient_age[y==1], 
            color='red', label='Malignant', alpha=0.5)
plt.scatter(tumor_size[y==0], patient_age[y==0], 
            color='blue', label='Benign', alpha=0.5)

# decision boundary using original scale w and unscaled tumor_size
tumor_range = np.linspace(0.5, 10, 100)
age_boundary = (-b - w[0]*tumor_range) / w[1]

plt.plot(tumor_range, age_boundary, color='green', label='Decision Boundary')
plt.ylim(20, 80)  # keep age in valid range

plt.xlabel("Tumor Size")
plt.ylabel("Age")
plt.title("Tumor Size vs Age")
plt.legend()
plt.show()
# get predictions on scaled x
f_x_final = 1/(1 + np.exp(-(x @ w + b)))


plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Decreasing")

plt.show()



#output with absolute max scaling
'''
[0.6656691  0.03370544]
-4.0679504610404695
0.40640706235467494
23377
'''

#============================================

#output with min max scaling
'''
[0.6656295  0.03414855]
-3.073698899892496
0.4063952981482353
16688
'''

#============================================


#Z-score scaling
'''
[1.90286268 0.59491739]
1.0694851863904424
0.4063742263375999
1819

No rescaling done
'''


