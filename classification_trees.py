import numpy as np

np.random.seed(42)
m = 500

study_hours = np.random.uniform(0, 10, m)
sleep_hours = np.random.uniform(4, 10, m)
attendance = np.random.uniform(0, 100, m)

# pass if studies enough AND decent attendance OR sleeps well and studies moderately
y = ((study_hours > 6) & (attendance > 75) |
     (sleep_hours > 7) & (study_hours > 4)).astype(int)

x = np.column_stack((study_hours, sleep_hours, attendance))

# shuffle first
indices = np.random.permutation(m)
x = x[indices]
y = y[indices]

# 80% for training, 20% for testing
split = int(0.8 * m)
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

y_test.astype(float)
y_train.astype(float)


def calc_gini(l1, l2):
     tot = l1[0] + l1[1] + l2[0] + l2[1]
     gini1 = 1-(((l1[0])/(l1[0]+l1[1]))**2) - (((l1[1])/(l1[0]+l1[1]))**2)
     gini2 = 1-(((l2[0])/(l2[0]+l2[1]))**2) - (((l2[1])/(l2[0]+l2[1]))**2)
     gini = ((gini1*(l1[0] + l1[1])) + (gini2*(l2[0] + l2[1])))/tot
     return gini


def find_condition(arr, y_sub):
     arr_org = arr
     arr = np.sort(arr)
     avg = np.zeros(len(arr)-1)
     for i in range(0,len(arr)-1):
          avg[i] = (arr[i]+arr[i+1])/2
     gini_imp = np.zeros(len(avg))
     k = 0
     for i in avg:
          l1 = np.array([0.0,0.0]) #[true, false]
          l2 = np.array([0.0,0.0])
          for j in range(0,int(len(arr_org))):
               if(arr_org[j] > i):
                    if(y_sub[j] == 1.0):
                         l1[0] += 1
                    else:
                         l1[1] += 1
               else:
                    if(y_sub[j] == 1.0):
                         l2[0] += 1
                    else:
                         l2[1] += 1
          
          gini_imp[k] = calc_gini(l1, l2)
          k += 1
     index = np.argmin(gini_imp)
     return np.array([avg[index], gini_imp[index]])


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
     if(len(set(y)) == 1):
          return int(y[0])

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
     if (tree == 0) or (tree == 1):
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


# FINAL TESTING
true = np.zeros(len(y_pred))
for i in range(0, len(y_pred)):
     if(y_pred[i] == y_test[i]):
          true[i] = 1
     else:
          true[i] = 0

print("The accuracy is:- ")
print((np.sum(true)/len(true))*100, "%")
# output
'''
The accuracy is:- 
99.0 %
'''



