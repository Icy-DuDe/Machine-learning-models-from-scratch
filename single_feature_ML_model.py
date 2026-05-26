import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)  # so results stay same every time

# 500 values for x
x = np.linspace(500, 3000, 500)

# true relationship
true_w = 0.1
true_b = 0

# add noise
noise = np.random.randn(500) * 10   # increase noise here

y = true_w * x + true_b + noise


w = 0
b = 0

w_history = []
b_history = []
cost_history = []


lr = 1e-8
diff = 10
i = 0

while(diff>1e-6 and i < 10000):

    error = ((w*(x) + b) - (y))
    cost_func = np.sum((error**2)/(2*(x.size)))

    dj_dw = np.sum((error*x)/(x.size))
    dj_db = np.sum((error)/(x.size))

    temp_w = w - lr*dj_dw
    temp_b = b - lr*dj_db
    
    diff = abs(w-temp_w)
    
    w = temp_w
    b = temp_b

    i += 1

    w_history.append(w)
    b_history.append(b)
    cost_history.append(cost_func)

print(w,"  -  ",b)


# plot actual data
plt.scatter(x, y, s = 3, alpha = 0.6)

# plot predicted line
y_pred = w*x + b
plt.plot(x, y_pred)

plt.xlabel("House Size")
plt.ylabel("Price")
plt.title("Final Model Fit")

plt.show()

plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Decreasing")

plt.show()

plt.scatter(x, y, s = 3, alpha = 0.6)

# early stage
plt.plot(x, w_history[0]*x + b_history[0], label="Start")

# mid stage
plt.plot(x, w_history[20]*x + b_history[20], label="Mid")

# final stage
plt.plot(x, w*x + b, label="Final")

plt.legend()
plt.show()
