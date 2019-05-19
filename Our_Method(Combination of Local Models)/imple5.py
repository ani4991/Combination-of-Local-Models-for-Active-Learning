from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.preprocessing import StandardScaler

# Loading the dataset
cancer_data = datasets.load_breast_cancer()
X_raw = cancer_data['data']
y_raw = cancer_data['target']
complete_data = np.insert(X_raw, 30, y_raw, axis=1)

# Standardize data
sc = StandardScaler()
sc.fit(X_raw)
X_raw = sc.transform(X_raw)

# Use RFE for feature selection
svm = LinearSVC(random_state=0, C=1, max_iter=100000)
rfe = RFE(svm, 20)
fit_cancer_data = rfe.fit(X_raw, y_raw)
transform_cancer_data = rfe.transform(X_raw)

# creating a dataframe out of numpy ndarrray
X_df = pd.DataFrame(X_raw)
X_df['y'] = cancer_data['target']
y_df = pd.DataFrame(y_raw)

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.3, random_state=0)

# Getting the best features
best_features = list()
a = 0
for i in rfe.ranking_:
    if i == 1:
        best_features.append(a)
    a += 1
best_features.append('y')

# Get list of all features
num_orig_feat = X_raw.shape[1]
orig_feat_index = list()
for i in range(num_orig_feat):
    orig_feat_index.append(i)

# Drop extra features ( original - best_features)
drop_features = list(set(orig_feat_index) - set(best_features))
X_train_reduced_feat = X_train.drop(drop_features, axis=1)
X_test_reduced_feat = X_test.drop(drop_features, axis=1)

# Get 60 samples with labels (30 of one class and 30 of other) from the training set
all_zero_labels = X_train_reduced_feat[X_train_reduced_feat['y'] == 0]
all_one_labels = X_train_reduced_feat[X_train_reduced_feat['y'] == 1]
subset_labeled_zero = all_zero_labels.drop(['y'], axis=1).head(50)
subset_labeled_one = all_one_labels.drop(['y'], axis=1).head(50)
subset_train_labeled_samples = pd.concat([subset_labeled_zero, subset_labeled_one])

# Get remaining samples without labels
remaining_labeled_zero = all_zero_labels.tail(all_zero_labels.shape[0] - subset_labeled_zero.shape[0])
remaining_labeled_one = all_one_labels.tail(all_one_labels.shape[0] - subset_labeled_one.shape[0])
remaining_train_labeled_samples = pd.concat([remaining_labeled_zero, remaining_labeled_one])
final_unlabeled_train_data = remaining_train_labeled_samples.drop(['y'], axis=1)
X_test_unlabeled = X_test_reduced_feat.drop(['y'], axis=1)

#creating list of tuples of the feature values in both classes 0 and 1
tuples_zero = [tuple(x) for x in subset_labeled_zero.values]
tuples_one = [tuple(x) for x in subset_labeled_one.values]

#renaming the columns of the dataframe in sequence[ 0,1,2....length of selected features]
new_column_name_training = list()
for num in range(len(best_features) - 1):
    new_column_name_training.append(num)
final_unlabeled_train_data.columns = new_column_name_training

new_column_name_X_test = list()
for num in range(len(best_features) - 1):
    new_column_name_X_test.append(num)
X_test_unlabeled.columns = new_column_name_X_test

# creating the final unlabeled data from the one's that have not been picked for training samples along with the test data
final_unlabeled_train_data_tuples = [tuple(x) for x in final_unlabeled_train_data.values]
X_test_unlabeled_coords_tuples = [tuple(x) for x in X_test_unlabeled.values]
final_unlabeled_train_data['coords'] = final_unlabeled_train_data_tuples
X_test_unlabeled['coords'] = X_test_unlabeled_coords_tuples

# Get distance of opposite labels
subset_labeled_zero['Coord'] = tuples_zero
subset_labeled_one['Coord'] = tuples_one
zero_index_values = list(subset_labeled_zero.index.values)
one_index_values = list(subset_labeled_one.index.values)

# We need to calculate the distance between the opposite labels and store it.
row_values = pd.Index(zero_index_values, name='rows')
column_values = pd.Index(one_index_values, name='columns')
dist_matrix = pd.DataFrame(data=0, index=row_values, columns=column_values)

#picking the closest - oppositely labeld instances
for row, column in subset_labeled_zero.iterrows():
    for row2, column2 in subset_labeled_one.iterrows():
        sum_squares = 0
        for i in range(len(best_features) - 1):
            sum_squares += (column['Coord'][i] - column2['Coord'][i]) ** 2
        dist_matrix.loc[row, row2] = sqrt(sum_squares)

# Get min distance value for each row. Problem: may not pick best min global value for each row. i.e: row 1 may pick
# column 4 as its min value but it turns out that column 4 and row 3 were closer but row 1 got to pick first.
# NOTE: mean doesn't work well for k-neighbors, using midpoint instead.

min_dist_list = list()
for row3, column3 in dist_matrix.iterrows():
    v = dist_matrix.values
    global_min_dist_index = np.nanargmin(v)
    i, j = [x[0] for x in np.unravel_index([global_min_dist_index], v.shape)]
    zero_one_index_pair = (dist_matrix.index[i], dist_matrix.columns[j])
    min_dist_list.append(zero_one_index_pair)
    dist_matrix.loc[zero_one_index_pair[0], zero_one_index_pair[1]] = np.nan

# Find midpoint to choose k-nearest neighbor.
midpoints = list()
get_features = list(set(best_features) - set(list(['y'])))
for dist_pair in min_dist_list:
    zero_label_coord = subset_labeled_zero.loc[dist_pair[0]]
    one_label_coord = subset_labeled_one.loc[dist_pair[1]]
    mp = list()
    for i in get_features:
        mp.append((zero_label_coord[i] + one_label_coord[i]) / 2)
    midpoints.append(tuple(mp))

#renaming the columns for zero-labeld subset as well as one-labeled subset
renamed_subset_zero = subset_labeled_zero
new_column_name_zero = list()
for num in range(len(best_features) - 1):
    new_column_name_zero.append(num)
new_column_name_zero.append('Coord')
renamed_subset_zero.columns = new_column_name_zero

renamed_subset_one = subset_labeled_one
new_column_name_one = list()
for num in range(len(best_features) - 1):
    new_column_name_one.append(num)
new_column_name_one.append('Coord')
renamed_subset_one.columns = new_column_name_one

# Find 2 nearest zero neighbors to each midpoint
k_nearest_zeros = dict()
for point_zero in midpoints:
    for (a, b) in min_dist_list:
        k_zero_points_dist = list()
        for index, row in subset_labeled_zero.iterrows():
            if index != a:
                v = 0
                for i in range(len(best_features) - 1):
                    v += ((row[i] - point_zero[i]) ** 2)
                k_zero_points_dist.append((sqrt(v), index))
        sorted_list = sorted(k_zero_points_dist, key=lambda x: x[0])
        k_nearest_zeros[point_zero] = sorted_list[:2]

# Find 2 nearest one neighbors to each midpoint
k_nearest_ones = dict()
for point_one in midpoints:
    for (a, b) in min_dist_list:
        k_one_points_dist = list()
        for index, row in subset_labeled_one.iterrows():
            if index != a:
                u = 0
                for j in range(len(best_features) - 1):
                    u += ((row[j] - point_one[j]) ** 2)
                k_one_points_dist.append((sqrt(u), index))
        sorted_list = sorted(k_one_points_dist, key=lambda x: x[0])
        k_nearest_ones[point_one] = sorted_list[:2]


# Combine zeros and one labels
combine_nearest_neigh = [k_nearest_zeros, k_nearest_ones]
nearest_neigh = dict()
for k in k_nearest_zeros.keys():
    nearest_neigh[k] = tuple(nearest_neigh[k] for nearest_neigh in combine_nearest_neigh)

# Create logistic regression model
local_classifiers = dict()
for points in midpoints:
    zero_neighbors = list()
    one_neighbors = list()
    zero_values_midpoint = k_nearest_zeros.get(points)
    one_values_midpoint = k_nearest_ones.get(points)
    zero_neighbors.append([b for (a, b) in zero_values_midpoint])
    one_neighbors.append([b for (a, b) in one_values_midpoint])
    all_neighbors_zero = subset_labeled_zero.loc[zero_neighbors[0], :]
    all_neighbors_one = subset_labeled_one.loc[one_neighbors[0], :]
    all_neighbors = pd.concat([all_neighbors_zero, all_neighbors_one])
    X_neighbors_data = all_neighbors.to_numpy()[:4, :(len(best_features) - 1)]
    y_neighbors_labels = list()
    for index, row in all_neighbors.iterrows():
        y_neighbors_labels.append(X_train.loc[index]['y'])
    y_neighbors_data = np.asarray(y_neighbors_labels)
    #creating the logistic regressor local model for each Midpoint
    classifier = LogisticRegression(C=1, random_state=0, solver='lbfgs', max_iter=1000)
    classifier.fit(X_neighbors_data, y_neighbors_data)
    local_classifiers[points] = classifier


# For each data point in train set calculate the distance to each midpoint
# TODO: Rename this variable to complete_train_data??
complete_test_data = pd.concat([final_unlabeled_train_data, X_test_unlabeled])
test_midpoint_least_dist_list = list()

#store the nearest Midpoint for every test data
for index, row in complete_test_data.iterrows():
    test_midpoint_list = list()
    for point in midpoints:
        dist = 0
        for i in range(len(best_features) - 1):
            dist += ((row[i] - point[i]) ** 2)
        test_midpoint_list.append((sqrt(dist), row['coords'], index, point))
    sorted_list = sorted(test_midpoint_list, key=lambda x: x[0])
    test_midpoint_least_dist_list.append(sorted_list[0][0:4])


# Iterating over the list(test_midpoint_least_dist_list) of info as previously mentioned
# matching the right local model by equating the values of mid points in the inner for-loop
# storing the result of the prediction of all the test points in a list(y_pred)
y_pred = list()
num_queries = 0
query_constraint = 20
final_remaining_unlabeled = test_midpoint_least_dist_list.copy()

for (dist, test_points, ind, mid_points) in test_midpoint_least_dist_list:
    closest_points = nearest_neigh[mid_points]
    zero_distances = [a[0] for a in closest_points[0]]
    one_distances = [b[0] for b in closest_points[1]]
    dist_clos_pts = zero_distances + one_distances
    min_dist_val = min(dist_clos_pts)
    if dist < min_dist_val and num_queries < query_constraint:
        # Ask annotator
        y_label_test_point = X_df.loc[ind]['y']
        element_to_remove = 0
        for item in test_midpoint_least_dist_list:
            if item[2] == ind:
                element_to_remove = item
        # index_elem_to_rem = test_midpoint_least_dist_list.index(element_to_remove)
        final_remaining_unlabeled.remove(element_to_remove)
        X_test_pnt_label = X_df.loc[ind][best_features[:len(best_features)-1]]
        X_test_coord = tuple(x[1] for x in X_test_pnt_label.iteritems())
        rename_indexes = [i for i in range(len(best_features) - 1)]
        X_test_pnt_label.index = rename_indexes
        X_test_pnt_label['Coord'] = X_test_coord
        num_queries += 1

        current_local_model = local_classifiers[mid_points]
        if y_label_test_point == 1 or y_label_test_point == 1.0:
            k_nearest_ones[mid_points].append((dist, ind))
            subset_labeled_one = subset_labeled_one.append(X_test_pnt_label)
        else:
            k_nearest_zeros[mid_points].append((dist, ind))
            subset_labeled_zero = subset_labeled_zero.append(X_test_pnt_label)

        zero_neighbors = list()
        one_neighbors = list()
        zero_values_midpoint = k_nearest_zeros.get(mid_points)
        one_values_midpoint = k_nearest_ones.get(mid_points)
        zero_neighbors.append([b for (a, b) in zero_values_midpoint])
        one_neighbors.append([b for (a, b) in one_values_midpoint])

        all_neighbors_zero = subset_labeled_zero.loc[zero_neighbors[0], :]
        all_neighbors_one = subset_labeled_one.loc[one_neighbors[0], :]
        all_neighbors = pd.concat([all_neighbors_zero, all_neighbors_one])
        X_neighbors_data = all_neighbors.to_numpy()[:all_neighbors.shape[0], :(len(best_features) - 1)]
        y_neighbors_labels = list()
        for index, row in all_neighbors.iterrows():
            y_neighbors_labels.append(X_df.loc[index]['y'])
        y_neighbors_data = np.asarray(y_neighbors_labels)

        current_local_model.fit(X_neighbors_data, y_neighbors_data)
        local_classifiers[mid_points] = current_local_model

# for other test data that is certain or the number of queries has reached its limit, we use the local classifier to get the label
for (dist, test_points, ind, mid_points) in final_remaining_unlabeled:
    for key in local_classifiers:
        ctr = 0
        for i in range(len(key)):
            if mid_points[i] == key[i]:
                ctr += 1
        if ctr == len(key):
            m = local_classifiers[key]
            vect = np.asarray([test_points])
            pred = m.predict(vect)
            y_pred.append((pred, ind))
            break


unlabeled_y_labels = remaining_train_labeled_samples['y']
final_unlabeld_y_labels = pd.concat([unlabeled_y_labels, y_test])

# Preparing test data and predicted data for accuracy
y_final_pred = list()  # List of predicted labels
for (i, j) in y_pred:
    y_final_pred.append((int(i[0]), j))

y_final_test = list()  # List of true labels
for (dist, test_points, ind, mid_points) in final_remaining_unlabeled:
    y_label = X_df.loc[ind]['y']
    y_final_test.append((int(y_label), ind))

ctr = 0
for f, b in zip(y_final_test, y_final_pred):
    if f[1] == b[1] and f[0] == b[0]:
        ctr += 1

# Printing the accuracy
acc = float(ctr / len(y_final_test))
print("ACC: ", acc)
print("Number of Correctly classified: ", ctr)
print("LENGTH OF TOTAL SAMPLES: ", len(y_final_test))