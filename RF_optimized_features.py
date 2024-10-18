import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import open3d as o3d
from tqdm import tqdm
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

extracted_dir = './data/Extracted_features'
os.makedirs(extracted_dir, exist_ok=True)

projects = ['Project1', 'Project2', 'Project3', 'Project4']

all_features = pd.DataFrame()
all_labels = []

# Feature extraction for original data
for project in tqdm(projects, desc="Processing projects"):
    json_file = f'./data/Pre_processed/{project}_metadata.json'
    
    with open(json_file, "r") as jsonfile:
        metadata_list = json.load(jsonfile)
    
    for bb in tqdm(metadata_list, desc=f"Extracting features from {project}", leave=False):
        points = pd.read_csv(os.path.join('./data/Pre_processed/', bb["pc_filename"]),
                             header=None, names=['x', 'y', 'z', 'i', 'r', 'g', 'b'],
                             sep=" ", usecols=range(0, 7))
        
        features = extract_point_features(points, n_bins=512)
        all_features = pd.concat([all_features, features], ignore_index=True)
        all_labels.append(bb["label"])

print('Original feature extraction finished.')

# TODO: Integrate augmented data
json_file_path = f'./data/Augmented/augmented.json'

with open(json_file_path, "r") as jsonfile:
    aug_metadata_list = json.load(jsonfile)

for bb in tqdm(aug_metadata_list, desc="Extracting augmented features"):
    points = pd.read_csv(os.path.join('./data/Augmented/', bb["pc_filename"]),
                         header=None, names=['x', 'y', 'z', 'i', 'r', 'g', 'b'],
                         sep=" ", usecols=range(0, 7))

    features = extract_point_features(points, n_bins=512)
    all_features = pd.concat([all_features, features], ignore_index=True)
    all_labels.append(bb["label"])

print('Augmented feature extraction finished.')

# Save all_features and all_labels locally
features_path = os.path.join(extracted_dir, 'all_features.csv')
labels_path = os.path.join(extracted_dir, 'all_labels.csv')

all_features.to_csv(features_path, index=False)
pd.DataFrame(all_labels, columns=['label']).to_csv(labels_path, index=False)

print(f"Features saved to: {features_path}")
print(f"Labels saved to: {labels_path}")

X = all_features
y = all_labels

# Load saved data
all_features = pd.read_csv(features_path)
all_labels = pd.read_csv(labels_path)['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training RandomForestClassifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# test_results = pd.DataFrame({
#     'Actual': y_test,
#     'Predicted': y_pred
# })

# print(test_results)

print(classification_report(y_test, y_pred))




