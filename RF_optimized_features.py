import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import open3d as o3d
import json
import os

def extract_point_features(points, n_bins = 256):
    center = points[['x', 'y', 'z']].mean()
    points[['x', 'y', 'z']] -= center
    
    features = []
    for feature in ['x', 'y', 'z', 'i', 'r', 'g', 'b']:
        # Calculate histogram for the current feature
        hist, _ = np.histogram(points[feature], bins=n_bins, density=True)
        # Normalize the histogram to get probabilities
        hist_prob = hist / hist.sum()
        features.extend(hist_prob)

    # Note: Coumputing normal vectors is very time-consuming. 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points)[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(np.array(points)[:, 4:]/256)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    # Convert normals to numpy array
    normals = np.asarray(pcd.normals)
    normals_df = pd.DataFrame(normals, columns=['normal_x', 'normal_y', 'normal_z'])
    
    # Calculate histograms for normal_x, normal_y, normal_z
    for normal_feature in ['normal_x', 'normal_y', 'normal_z']:
        hist, _ = np.histogram(normals_df[normal_feature], bins=n_bins, density=True)
        hist_prob = hist / hist.sum()
        features.extend(hist_prob)

    # Convert features list to a numpy array
    feature_vector = np.array(features) # (#bin * 10,)
    # print("features shape before:", feature_vector.shape)
    features_df = pd.DataFrame([features]) # (1, #bin * 10)
    # print("features shape after:", features_df.shape)
    return features_df

projects = ['Project1', 'Project2', 'Project3', 'Project4']

all_features = pd.DataFrame()
all_labels = []

for project in projects:
    json_file = f'./data/Pre_processed/{project}_metadata.json'

    with open(json_file, "r") as jsonfile:
            metadata_list = json.load(jsonfile)

    for bb in metadata_list:
        points = pd.read_csv(os.path.join('./data/Pre_processed/', bb["pc_filename"]),
                    header=None, names=['x', 'y', 'z', 'i', 'r', 'g', 'b'],
                    sep=" ", usecols=range(0, 7))

        features = extract_point_features(points, n_bins = 512)
        all_features = pd.concat([all_features, features], ignore_index=True)
        all_labels.append(bb["label"])

    print("all_features shape:", all_features.shape)
    # print("Feature vector [0]:", all_features.iloc[0, :])


X = all_features
y = all_labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# test_results = pd.DataFrame({
#     'Actual': y_test,
#     'Predicted': y_pred
# })

# print(test_results)

print(classification_report(y_test, y_pred))




