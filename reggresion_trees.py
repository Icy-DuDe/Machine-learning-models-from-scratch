import numpy as np

np.random.seed(42)
m = 500

size = np.random.uniform(500, 4000, m)        # sq feet
rooms = np.random.randint(1, 6, m).astype(float)
age = np.random.uniform(0, 50, m)             # house age in years

noise = np.random.randn(m) * 10000

price = 50*size + 20000*rooms - 1000*age + 100000 + noise

x = np.column_stack((size, rooms, age))
y = price

# shuffle first
indices = np.random.permutation(m)
x = x[indices]
y = y[indices]

# 80% for training, 20% for testing
split = int(0.8 * m)
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

def calc_residuals(average, y_sub):
    res = 0
    for i in range(0, len(y_sub)):
        res += (y_sub[i] - average)**2
    return res

def find_condition(arr, y_sub):
    arr_org = arr
    arr = np.sort(arr)
    k = 0
    avg = (arr[:-1] + arr[1:]) / 2
    res = np.zeros(len(avg))
    for i in avg:                   
        right_arr = arr_org > i
        left_arr = arr_org <= i
        if np.sum(right_arr) == 0 or np.sum(left_arr) == 0:
            res[k] = float('inf')  # penalize empty splits
            k += 1
            continue
        right_avg = np.mean(y_sub[right_arr])
        left_avg = np.mean(y_sub[left_arr])
        res[k] = calc_residuals(right_avg, y_sub[right_arr]) + calc_residuals(left_avg, y_sub[left_arr])
        k += 1
    order = np.argsort(res)
    return np.array([avg[order[0]], res[order[0]]])


def find_best_split(x1, x2, x3, y):
    gini = np.zeros((3, 2))
    gini[0] = find_condition(x1, y)
    gini[1] = find_condition(x2, y)
    gini[2] = find_condition(x3, y)

    order = np.argsort(gini[:, 1])
    best_feature = order[0]
    best_threshold = gini[order[0], 0]
    
    return int(best_feature), float(best_threshold)

def build_tree(x, y):
     if(len(y) <= 5):
         return float(np.mean(y))  
    

     x1 = x[:,0]
     x2 = x[:,1]
     x3 = x[:,2]
     best_feature, best_threshold = find_best_split(x1, x2, x3, y)


     right_mask = x[:, best_feature] > best_threshold
     left_mask = ~right_mask  # opposite
     right_x = x[right_mask]
     right_y = y[right_mask]
     left_x = x[left_mask]
     left_y = y[left_mask]
     
     right_branch = build_tree(right_x, right_y)
     left_branch = build_tree(left_x, left_y)

     return (best_feature, best_threshold, left_branch, right_branch)


def predict(tree, x_test):
     if (type(tree) == float):
          return tree
     else:
          feature, threshold, left_branch, right_branch = tree
          if (x_test[feature] > threshold):
               return predict(right_branch, x_test)
          else:
               return predict(left_branch, x_test)
          
tree = build_tree(x_train, y_train)

y_pred = np.zeros(int(0.2*m))

for i in range(0, len(y_pred)):
     y_pred[i] = predict(tree, x_test[i])

#Final checking:-

mae = np.mean(np.abs(y_pred - y_test))
print(f"MAE: ${mae:.2f}")

