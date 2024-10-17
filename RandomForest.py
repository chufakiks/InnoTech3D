import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def RandomForest(voxel_size=None):
    all_points = []
    all_boxes = []

    for project in ['Project1', 'Project2', 'Project3', 'Project4']:

        # Columns contain x, y, z, i, r, g, b values
        s_points = pd.read_csv("data/TrainingSet/{}/{}.xyz".format(project, project), header=None, names=['x', 'y', 'z', 'i', 'r', 'g', 'b'])
        s_points['project'] = project
        all_points.append(s_points)

        s_boxes = pd.read_csv('data/TrainingSet/{}/{}.csv'.format(project, project))
        s_boxes['project'] = project  # Add project identifier
        all_boxes.append(s_boxes)
        print('The number of boxes of {} is {}'.format(project, len(s_boxes)))

    points = pd.concat(all_points, ignore_index=True)
    boxes = pd.concat(all_boxes, ignore_index=True)
    print('Total boxes combined'.format(len(boxes)))

    boxes.columns = boxes.columns.str.strip()

    points[['x', 'y', 'z', 'i', 'r', 'g', 'b']] = points[['x', 'y', 'z', 'i', 'r', 'g', 'b']].apply(pd.to_numeric,
                                                                                                    errors='coerce')
    boxes[['BB.Min.X', 'BB.Min.Y', 'BB.Min.Z', 'BB.Max.X', 'BB.Max.Y', 'BB.Max.Z']] = boxes[
        ['BB.Min.X', 'BB.Min.Y', 'BB.Min.Z', 'BB.Max.X', 'BB.Max.Y', 'BB.Max.Z']].apply(pd.to_numeric, errors='coerce')

    points = points.dropna()
    boxes = boxes.dropna()

    point_features = boxes.apply(lambda row: extract_point_features(row, points), axis=1)

    # Combine bounding box geometric features and point cloud features.
    boxes['length'] = boxes['BB.Max.X'] - boxes['BB.Min.X']
    boxes['width'] = boxes['BB.Max.Y'] - boxes['BB.Min.Y']
    boxes['height'] = boxes['BB.Max.Z'] - boxes['BB.Min.Z']
    combined_features = pd.concat([boxes[['length', 'width', 'height']], point_features], axis=1)

    # Converting column names to string types to ensure scikit-learn compatibility
    combined_features.columns = combined_features.columns.astype(str)

    X = combined_features
    y = boxes['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    return


# feature extraction
def extract_point_features(box, points):

    # bounding box masks
    mask = (
            (points['x'] >= box['BB.Min.X']) & (points['x'] <= box['BB.Max.X']) &
            (points['y'] >= box['BB.Min.Y']) & (points['y'] <= box['BB.Max.Y']) &
            (points['z'] >= box['BB.Min.Z']) & (points['z'] <= box['BB.Max.Z'])
    )
    selected_points = points[mask]

    if len(selected_points) == 0:
        return pd.Series([0] * 10)

    # points features
    feature_vector = [
        len(selected_points),  
        selected_points[['x', 'y', 'z']].mean().values,  
        selected_points[['x', 'y', 'z']].std().values,  
        selected_points['i'].mean(),  
        selected_points[['r', 'g', 'b']].mean().values,  
    ]
    return pd.Series(np.concatenate(feature_vector))

if __name__ == "__main__":
    RandomForest()

