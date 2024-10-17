# 1. rotation matrix
# 2. drop ouf
# TO-DO: add the cor of bounding box into each point cloud. x_min, 

# mdkir "./Augmented"

# for each_.xyz_file in "./Pre_processed":
#     # filename = "{} {} {}.xyz".format(project, label, id)
#     # read .xyz file to get 
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


    for project in ['Project1', 'Project2', 'Project3', 'Project4']:
        json_path = "./data/Pre_processed/{}_metadata.json".format(project)
        calculate_label(json_path, label_counts)
        # augment(json_path)

    # Print out the label frequencies
    print("Label Frequencies across all projects:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

def calculate_label(json_path, label_counts):
    
    with open(json_path, "r") as jsonfile:
        metadata_list = json.load(jsonfile)

    for bb in metadata_list:
        # Count the label occurrence
        label = bb["label"]
        if label in label_counts:
            label_counts[label] += 1
        label_counts["Overall"] += 1


def augment(json_path, dp_rate = 0.2):
    with open(json_path, "r") as jsonfile:
        metadata_list = json.load(jsonfile)
    
    # i = 0

    for bb in metadata_list:
        
        # if i > 1:
        #     break
        # i = i + 1

        project = bb["project"]
        label = bb["label"]
        bb_min = np.array([bb["BB.Min.X"], bb["BB.Min.Y"], bb["BB.Min.Z"]])
        bb_max = np.array([bb["BB.Max.X"], bb["BB.Max.Y"], bb["BB.Max.Z"]])
        
        # Columns contain x, y, z, i, r, g, b values
        xyz = pd.read_csv(os.path.join('./data/Pre_processed/', bb["pc_filename"]),
                            sep=" ", usecols=range(0, 7), header=None)

        # Pass xyz and rgb to Open3D.o3d.geometry.PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(xyz)[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(np.array(xyz)[:, 4:]/256)
        
        # Define the axis-aligned bounding box
        aabb = o3d.geometry.AxisAlignedBoundingBox(bb_min, bb_max)
        # Convert AABB to an oriented bounding box
        obb = aabb.get_oriented_bounding_box()
        obb.color = [0, 1, 0]
        
        # obb_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=obb.center)

        # Random rotation matrix
        R = pcd.get_rotation_matrix_from_xyz(np.random.uniform(0, 2 * np.pi, 3))
        R_center = np.random.uniform(-5, 5, 3)

        # Apply the Rotation to the point cloud
        pcd_aug = copy.deepcopy(pcd)
        pcd_aug.rotate(R, center=R_center)

        # Apply the same Rotation to the oriented bounding box
        obb_aug = copy.deepcopy(obb)
        obb_aug.rotate(R, center=R_center)
        obb_aug.color = [1, 0, 0]
        
        # # Apply the same Rotation to the coordinate frames
        # obb_aug_frame = copy.deepcopy(obb_frame)
        # obb_aug_frame.rotate(R, center=R_center)
        # obb_aug_frame.scale(4, center=obb_aug_frame.get_center())

        # Dropout
        indices_to_keep = np.random.choice(len(pcd_aug.points), int((1 - dp_rate) * len(pcd_aug.points)), replace=False)
        pcd_dropped = pcd_aug.select_by_index(indices_to_keep)

        # # # Visualize original and transformed point clouds and bounding boxes
        # o3d.visualization.draw_geometries([pcd, obb, pcd_dropped, obb_aug])

if __name__ == "__main__":
    main()
