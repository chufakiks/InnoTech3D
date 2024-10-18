import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json
import os

def pre_process(project):
    point_cloud_file = 'data/HiddenSet/{}/{}.xyz'.format(project, project)
    bounding_box_file = 'data/HiddenSet/{}/{}.csv'.format(project, project)

    # Load point cloud data and bounding box data
    points = pd.read_csv(point_cloud_file, header=None, names=['x', 'y', 'z', 'i', 'r', 'g', 'b'], delimiter=' ')
    boxes = pd.read_csv(bounding_box_file, skipinitialspace=True)

    # Check column names and remove any potential spaces
    boxes.columns = boxes.columns.str.strip()

    # Convert columns involved in comparisons to numeric type
    points[['x', 'y', 'z', 'i', 'r', 'g', 'b']] = points[['x', 'y', 'z', 'i', 'r', 'g', 'b']].apply(pd.to_numeric, errors='coerce')
    boxes[['BB.Min.X', 'BB.Min.Y', 'BB.Min.Z', 'BB.Max.X', 'BB.Max.Y', 'BB.Max.Z']] = boxes[
        ['BB.Min.X', 'BB.Min.Y', 'BB.Min.Z', 'BB.Max.X', 'BB.Max.Y', 'BB.Max.Z']].apply(pd.to_numeric, errors='coerce')

    # Handle missing values (optional, depending on the situation)
    # points = points.dropna()
    # boxes = boxes.dropna()
    # print(f"boxes shape: {boxes.shape}")
    # print(boxes.head())
    # print(boxes.iloc[1])

    metadata_list = []
    point_features = boxes.apply(lambda row: extract_point_features(row, points, project, metadata_list), axis=1)
    save_metadata_json(metadata_list, project)


# Define a function to get the points within each bounding box and extract features
def extract_point_features(box, points, project, metadata_list):
    directory = './data/Hidden_Pre_processed/'
    os.makedirs(directory, exist_ok=True)
    # Filter points that are inside the bounding box
    mask = (
            (points['x'] >= box['BB.Min.X']) & (points['x'] <= box['BB.Max.X']) &
            (points['y'] >= box['BB.Min.Y']) & (points['y'] <= box['BB.Max.Y']) &
            (points['z'] >= box['BB.Min.Z']) & (points['z'] <= box['BB.Max.Z'])
    )
    selected_points = points[mask]
    # label = str(box['Label']) 
    id = str(box['ID'] )
    file_path = os.path.join(directory, "{} {}.xyz".format(project, id))
    xyzfile = open(file_path, "w")

    # Write the selected points to the file
    selected_points.to_csv(xyzfile, sep=' ', index=False, header=False)
    xyzfile.close()
    print("{}, points inside: {}".format(project, len(selected_points)))

    metadata_list.append({
        "project": project, 
        # "label": label, 
        "BB.Min.X": box['BB.Min.X'],
        "BB.Min.Y": box['BB.Min.Y'],
        "BB.Min.Z": box['BB.Min.Z'],
        "BB.Max.X": box['BB.Max.X'],
        "BB.Max.Y": box['BB.Max.Y'],
        "BB.Max.Z": box['BB.Max.Z'],
        "id": id,
        "pc_filename": f"{project} {id}.xyz"
    })


# Save the entire metadata list for the project into a single JSON file
def save_metadata_json(metadata_list, project):
    directory = './data/Hidden_Pre_processed/'
    json_file_path = os.path.join(directory, f"{project}_metadata.json")
    
    with open(json_file_path, "w") as jsonfile:
        json.dump(metadata_list, jsonfile, indent=4)
    
    print(f"Metadata JSON for {project} saved as: {json_file_path}")


def main():
    for project in ['Project1', 'Project2', 'Project3', 'Project4']:
        pre_process(project)


if __name__ == "__main__":
    main()