import numpy as np

# Set seed for reproducibility so you get the same numbers every time you run it
np.random.seed(42)

# Number of patients
n_samples = 400

# Feature 1: Age (Years)
age = np.random.normal(55, 10, n_samples)

# Feature 2: Tumor Size (cm)
tumor_size = np.random.normal(3.5, 1.5, n_samples)

# Feature 3: Cell Density (Scale 1-10)
cell_density = np.random.uniform(1, 10, n_samples)

# Feature 4: Family History (Binary: 0 or 1)
# 30% chance of having a family history
family_history = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

# Create the Target Variable (0 = Benign, 1 = Malignant)
# We create a hidden mathematical rule that the Random Forest needs to learn.
# Malignant tumors are more likely if: size is large AND density is high, or if there's family history
probabilities = (
    0.3 * (tumor_size > 4.0) + 
    0.4 * (cell_density > 6.5) + 
    0.2 * family_history + 
    0.1 * (age > 60)
)

# Add some random noise so it's not a perfectly clean split (forces the forest to work hard)
probabilities += np.random.normal(0, 0.1, n_samples)

# Convert probabilities to strictly 0 or 1
y = np.where(probabilities > 0.5, 1, 0)

# Combine all features into an X matrix (Shape: 400 rows, 4 columns)
X = np.column_stack((age, tumor_size, cell_density, family_history))

print("Dataset created successfully!")
print(f"Shape of X (Features): {X.shape}")
print(f"Shape of y (Labels): {y.shape}")
print(f"Number of Malignant cases: {np.sum(y == 1)}")
print(f"Number of Benign cases: {np.sum(y == 0)}")

# ++++++++++++++++++++++++++++++++===========================

# A different seed ensures these are 100 entirely new, unseen patients
np.random.seed(43)

n_test_samples = 100

# Feature 1: Age (Years)
age_test = np.random.normal(55, 10, n_test_samples)

# Feature 2: Tumor Size (cm)
tumor_size_test = np.random.normal(3.5, 1.5, n_test_samples)

# Feature 3: Cell Density (Scale 1-10)
cell_density_test = np.random.uniform(1, 10, n_test_samples)

# Feature 4: Family History (Binary: 0 or 1)
family_history_test = np.random.choice([0, 1], size=n_test_samples, p=[0.7, 0.3])

# The exact same hidden rule the forest was trained to find
probabilities_test = (
    0.3 * (tumor_size_test > 4.0) + 
    0.4 * (cell_density_test > 6.5) + 
    0.2 * family_history_test + 
    0.1 * (age_test > 60)
)

# Add the same random noise
probabilities_test += np.random.normal(0, 0.1, n_test_samples)

# Convert to 0 or 1 labels
y_test = np.where(probabilities_test > 0.5, 1, 0)

# Combine into the final X_test matrix
X_test = np.column_stack((age_test, tumor_size_test, cell_density_test, family_history_test))

print("Test dataset created successfully!")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_test: {y_test.shape}")

# ++++++++++++++++++++++++++++++++===========================


class Trees:

    random_feature_no = 2


    def __init__(self, arr, y):
        boot_arr = np.zeros((400,4))
        OOB_arr = np.arange(0, 400)
        y_boot_arr = np.zeros(400)
        
        for i in range(0, 400):
            j = np.random.randint(0,400)
            OOB_arr = OOB_arr[OOB_arr != j]
            boot_arr[i] = arr[j]
            y_boot_arr[i] = y[j]
        
        self.OOB = OOB_arr

        self.tree = Trees.build_tree(boot_arr, y_boot_arr)

    def build_tree(x, y):
        # Quick safety check: if y is empty or pure, return a leaf
        if len(y) == 0: 
            return 0
        if len(set(y)) == 1:
            return int(y[0])
        
        selection = np.random.choice([0, 1, 2, 3], size=Trees.random_feature_no, replace=False)
        x_new = x[:, selection]
        best_feature, best_threshold = Trees.find_best_split(x_new, y)

        # 1. MAP BACK TO GLOBAL FEATURE INDEX (Add this line)
        actual_best_feature = selection[best_feature]

        # 2. USE THE ORIGINAL 'x' INSTEAD OF 'x_new' TO PRESERVE COLUMNS
        right_mask = x[:, actual_best_feature] > best_threshold
        left_mask = ~right_mask  
        
        right_x = x[right_mask]
        right_y = y[right_mask]
        left_x = x[left_mask]
        left_y = y[left_mask]

        # === ADD THIS SAFETY CHECK HERE ===
        # If a split puts ALL data into one side, it's a dead end. Stop and make a leaf!
        if len(left_y) == len(y) or len(right_y) == len(y):
            # Take a majority vote of the local labels at this node
            return 1 if np.sum(y == 1) >= np.sum(y == 0) else 0
        # ==================================
        
        right_branch = Trees.build_tree(right_x, right_y)
        left_branch = Trees.build_tree(left_x, left_y)

        # 3. RETURN THE ACTUAL GLOBAL FEATURE INDEX
        return (actual_best_feature, best_threshold, left_branch, right_branch)

    
    def find_best_split(x, y):
        gini = np.zeros((len(x[0]), 2))
        for i in range(0, len(x[0])):
            gini[i] = Trees.find_condition(x[:,i], y)


        order = np.argsort(gini[:, 1])
        best_feature = order[0]
        best_threshold = gini[order[0], 0]
    
        return int(best_feature), float(best_threshold)

    def find_condition(arr, y_sub):
        # 1. Sort the feature array and keep the labels aligned
        sort_indices = np.argsort(arr)
        sorted_arr = arr[sort_indices]
        sorted_y = y_sub[sort_indices]

        # 2. Calculate all midpoints (thresholds) instantly
        # Using array slicing: (left_elements + right_elements) / 2
        avg = (sorted_arr[:-1] + sorted_arr[1:]) / 2.0

        # 3. Use cumulative sums to count 1s and 0s on the left of EVERY threshold
        # Since y_sub only contains 1.0 and 0.0, the sum is exactly the count of 1s.
        left_1_count = np.cumsum(sorted_y)[:-1] 
        left_total = np.arange(1, len(sorted_y))
        left_0_count = left_total - left_1_count

        # 4. Calculate counts on the right side by subtracting left from total
        total_1s = np.sum(sorted_y)
        total_0s = len(sorted_y) - total_1s
        
        right_1_count = total_1s - left_1_count
        right_0_count = total_0s - left_0_count
        right_total = len(sorted_y) - left_total

        # 5. Calculate Gini Impurity for all thresholds simultaneously
        # Gini = 1 - (prob_1^2 + prob_0^2)
        gini_left = 1.0 - ((left_1_count / left_total)**2 + (left_0_count / left_total)**2)
        gini_right = 1.0 - ((right_1_count / right_total)**2 + (right_0_count / right_total)**2)

        # Weighted average of the two sides
        n_total = len(sorted_y)
        weighted_gini = (left_total / n_total) * gini_left + (right_total / n_total) * gini_right

        # 6. Find the index of the minimum Gini impurity
        index = np.argmin(weighted_gini)

        return np.array([avg[index], weighted_gini[index]])
    
    def predict(tree, x_test):
     if (tree == 0) or (tree == 1):
          return tree
     else:
          feature, threshold, left_branch, right_branch = tree
          if (x_test[feature] > threshold):
               return Trees.predict(right_branch, x_test)
          else:
               return Trees.predict(left_branch, x_test)
            

OOB_error = np.zeros(3)

for i in [2, 3, 4]:
    Trees.random_feature_no = i

    random_forest = []
    OOB_pred = np.full((400, 100), -1)
    
    for j in range(0, 100):
        current_tree = Trees(X, y)
        random_forest.append(current_tree)
        OOB_array = current_tree.OOB

        for k in OOB_array:
            OOB_pred[k][j] = Trees.predict(current_tree.tree, X[k])

    oob_misses = 0
    total_evaluated_patients = 0

    for patient_idx in range(400):
        patient_votes = OOB_pred[patient_idx]
        valid_votes = patient_votes[patient_votes != -1]
        
        if len(valid_votes) > 0:
            total_evaluated_patients += 1
            
            count_1 = np.sum(valid_votes == 1)
            count_0 = np.sum(valid_votes == 0)
            
            majority_vote = 1 if count_1 > count_0 else 0
            
            if majority_vote != y[patient_idx]:
                oob_misses += 1

    # Store the error rate for this feature setting
    error_rate = (oob_misses / total_evaluated_patients) * 100
    OOB_error[i-2] = error_rate


    print(f"OOB Error with {i} features: {error_rate:.2f}%")

#===================================================

order = np.argsort(OOB_error)
num = order[0] + 2
random_forest = []

Trees.random_feature_no = num
for i in range(0, 100):
    tree = Trees(X,y)
    random_forest.append(tree.tree)

y_pred = np.zeros(100)

for i in range(0, 100):
    yes = 0
    no = 0
    for tree in random_forest:
        pred = Trees.predict(tree, X_test[i])
        if(pred == 1):
            yes += 1
        else:
            no += 1
    
    if(yes > no):
        y_pred[i] = 1
    else:
        y_pred[i] = 0

# FINAL TESTING
true = np.zeros(len(y_pred))
for i in range(0, len(y_pred)):
     if(y_pred[i] == y_test[i]):
          true[i] = 1
     else:
          true[i] = 0

print("The accuracy is:- ")
print((np.sum(true)/len(true))*100, "%")

# Output

'''
Dataset created successfully!
Shape of X (Features): (400, 4)
Shape of y (Labels): (400,)
Number of Malignant cases: 129
Number of Benign cases: 271
Test dataset created successfully!
Shape of X_test: (100, 4)
Shape of y_test: (100,)
OOB Error with 2 features: 10.50%
OOB Error with 3 features: 9.50%
OOB Error with 4 features: 10.00%
The accuracy is:- 
89.0 %
'''
