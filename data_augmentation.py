import pandas as pd
import open3d as o3d
import numpy as np
import json
import copy
import os

def main():

    np.random.seed(42)
    
    label_counts = {"Pipe": 0, "HVAC_Duct": 0, "Structural_IBeam": 0, "Structural_ColumnBeam": 0, "Overall": 0}
    aug_label_counts = {"Pipe": 0, "HVAC_Duct": 0, "Structural_IBeam": 0, "Structural_ColumnBeam": 0, "Overall": 0}
   
    min_threshold = 200  # Define your minimum threshold
    max_threshold = 250
    metadata_combined = []

    for project in ['Project1', 'Project2', 'Project3', 'Project4']:

        json_path = "./data/Pre_processed/{}_metadata.json".format(project)
        
        with open(json_path, "r") as jsonfile:
            metadata_list = json.load(jsonfile)
        
        metadata_combined.extend(metadata_list)
        # print(len(metadata_combined)) # =450

    calculate_label(metadata_combined, label_counts)
    print(label_counts)

    for label, count in label_counts.items():
        if label == "Overall":
            continue  # Skip the "Overall" key

        if count < min_threshold:
            # print(f"Augmenting {label} to reach {min_threshold}")
            augment(metadata_combined, target_label=label, required_count=min_threshold - count)
                
        elif min_threshold <= count < max_threshold:
            # print(f"Augmenting {label} to reach {max_threshold}")
            augment(metadata_combined, target_label=label, required_count=max_threshold - count)

        print(f'{label} Augment finished. ')

    augmented_json_path = "./data/Augmented/augmented.json"
    with open(json_path, "r") as jsonfile:
        aug_metadata_list = json.load(jsonfile)
    
    calculate_label(aug_metadata_list, aug_label_counts) # NOTE: Something wrong here!
    print(aug_label_counts)


def calculate_label(metadata_list, label_counts):

    print(f"Metadata list type: {type(metadata_list)}, length: {len(metadata_list)}")

    for bb in metadata_list:

        # Count the label occurrence
        label = bb["label"]
        if label in label_counts:
            label_counts[label] += 1
        label_counts["Overall"] += 1


def augment(json_list, target_label, required_count):

    print(f'target_label, {target_label}; required_count, {required_count}')

    augmented_dir = "./data/Augmented/"
    os.makedirs(augmented_dir, exist_ok=True)
    
    augmented_count = 0
    loop_time = 1

    while augmented_count < required_count:

        # loop obj in json_list and augment until meeting required_count. u can loop multiple times.
        for i, bb in enumerate(json_list):
            # Check if the object's label matches the target label
            if bb["label"] != target_label:
                continue

            # Perform and save each augmentation
            aug_and_save_helper('{}_{}'.format(loop_time, i), bb, augmented_dir)
            augmented_count += 1

            # Stop if the required augmentation count is met
            if augmented_count >= required_count:
                break
        
        loop_time += 1


def aug_and_save_helper(idx, bb, augmented_dir = "./data/Augmented/", dp_rate = 0.2):

    project = bb["project"]
    label = bb["label"]
    bb_min = np.array([bb["BB.Min.X"], bb["BB.Min.Y"], bb["BB.Min.Z"]])
    bb_max = np.array([bb["BB.Max.X"], bb["BB.Max.Y"], bb["BB.Max.Z"]])
    
    # Columns contain x, y, z, i, r, g, b values
    xyz = pd.read_csv(os.path.join('./data/Pre_processed/', bb["pc_filename"]),
                        sep=" ", usecols=range(0, 7), header=None, names=['x', 'y', 'z', 'i', 'r', 'g', 'b'])

    # Pass xyz and rgb to Open3D.o3d.geometry.PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[['x', 'y', 'z']].values)
    pcd.colors = o3d.utility.Vector3dVector(xyz[['r', 'g', 'b']].values / 256)
    # print(pcd.shape)
    
    # # Define the axis-aligned bounding box
    # aabb = o3d.geometry.AxisAlignedBoundingBox(bb_min, bb_max)
    # # Convert AABB to an oriented bounding box
    # obb = aabb.get_oriented_bounding_box()
    # obb.color = [0, 1, 0]
    
    # obb_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=obb.center)

    # Random rotation matrix
    R = pcd.get_rotation_matrix_from_xyz(np.random.uniform(0, 2 * np.pi, 3))
    R_center = np.random.uniform(-5, 5, 3)
    pcd.rotate(R, center=R_center)
    
    # Uncomment below to make comparision figures
    # # Apply the Rotation to the point cloud
    # pcd_aug = copy.deepcopy(pcd)
    # pcd_aug.rotate(R, center=R_center)

    # # Apply the same Rotation to the oriented bounding box
    # obb_aug = copy.deepcopy(obb)
    # obb_aug.rotate(R, center=R_center)
    # obb_aug.color = [1, 0, 0]
    
    # # Apply the same Rotation to the coordinate frames
    # obb_aug_frame = copy.deepcopy(obb_frame)
    # obb_aug_frame.rotate(R, center=R_center)
    # obb_aug_frame.scale(4, center=obb_aug_frame.get_center())

    # Dropout
    indices_to_keep = np.random.choice(len(pcd.points), int((1 - dp_rate) * len(pcd.points)), replace=False)
    pcd_dropped = pcd.select_by_index(indices_to_keep)

    # # # Visualize original and transformed point clouds and bounding boxes
    # o3d.visualization.draw_geometries([pcd, obb, pcd_dropped, obb_aug])

    # Extract augmented data including x, y, z, i, r, g, b
    augmented_points = np.asarray(pcd_dropped.points)
    augmented_colors = (np.asarray(pcd_dropped.colors) * 256).astype(int)  # Convert colors back to original scale
    augmented_intensity = xyz['i'].iloc[indices_to_keep].values.reshape(-1, 1)

    # Combine all features into a single array
    augmented_data = np.hstack((augmented_points, augmented_intensity, augmented_colors))
    # print(augmented_data.shape)

    # Save augmented point cloud to file
    filename = f"{project} Aug {label} {idx}.xyz"
    filepath = os.path.join(augmented_dir, filename)
    pd.DataFrame(augmented_data, 
                columns=['x', 'y', 'z', 'i', 'r', 'g', 'b']).to_csv(filepath, sep=' ', header=False, index=False)

    # Save metadata
    # box_min = pcd_dropped.get_min_bound() # ???? We don't need it anymore.
    # box_max = pcd_dropped.get_max_bound() # 
    metadata = {
        "project": project,
        "label": label,
        "pc_filename": filename
    }
    
    augmented_file = os.path.join(augmented_dir, "augmented.json")

    if os.path.exists(augmented_file):
        with open(augmented_file, "r") as jsonfile:
            existing_data = json.load(jsonfile)
    else:
        existing_data = []
    existing_data.append(metadata)

    # Write all augmented metadata to JSON
    with open(augmented_file, "w") as jsonfile:
        json.dump(existing_data, jsonfile, indent=4)

if _name_ == "_main_":
    main()