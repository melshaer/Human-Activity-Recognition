from collections import defaultdict
from PIL import Image as PIL_Image
from pathlib import Path
from statistics import mean
from sklearn.feature_selection import chi2
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re

images_dir_path = Path('images/')
train_annot_file = 'train.json'

'''data_schema = {'images' : {'id': imgidx, 'file_name': img_filename, 'width': w, 'height': h} ,
                  'activities' :  {'image_id': imgidx, 'activity_name': activity['act_name'].tolist() ,
                                    'category_name': activity['cat_name'].tolist(), 'activity_id': activity['act_id'].tolist()} ,
                  'annotations' : {'person_id': pid, 'image_id': imgidx, 'area': bbox[2]*bbox[3], 'bbox': bbox.tolist(), 
                                   'keypoints_x': annot_keypts[:, 0].tolist(), 'keypoints_y' : annot_keypts[:, 1].tolist(), 
                                   'num_keypoints': int(np.sum(keypts[:,2]==1))}}'''

'''keypoints_schema = ["r_ankle", "r_knee", "r_hip", "l_hip", "l_knee", "l_ankle", "pelvis", "throax", "upper_neck", "head_top", 
                          "r_wrist", "r_elbow", "r_shoulder", "l_shoulder", "l_elbow", "l_wrist"]'''

# Load data from disk
with open(train_annot_file) as f:
    data = json.load(f)


# Helper Functions
def get_activity_name(image_id):
    # Filter data to image_id only
    single_act_image = [entry for entry in data['activities'] if entry['image_id'] == image_id]
    return single_act_image[0]['activity_name'][0][0]


def get_category_name(image_id):
    # Filter data to image_id only
    single_act_image = [entry for entry in data['activities'] if entry['image_id'] == image_id]
    return single_act_image[0]['category_name'][0][0]


def get_annot(image_id):
    # Filter data to image_id only
    single_annot_image = [entry for entry in data['annotations'] if entry['image_id'] == image_id]
    return single_annot_image[0]


def get_num_persons(image_id):
    # Filter data to image_id only
    single_annot_image = [entry for entry in data['annotations'] if entry['image_id'] == image_id]
    return len(single_annot_image)


def get_image_id(activity_name):
    activity = [[activity_name]]
    image_id_list = [entry['image_id'] for entry in data['activities'] if entry['activity_name'] == activity]
    return image_id_list


def get_image_id_from_filename(filename):
    single_image = [entry for entry in data['images'] if entry['file_name'] == filename]
    return single_image[0]['id']


def get_image_filename_from_id(image_id):
    single_image = [entry for entry in data['images'] if entry['id'] == image_id]
    return single_image[0]['file_name']


# Display an example image with all annotated joints
def display(image_id):
    # Filter data to image_id only
    single_annot_image_data = [entry for entry in data['annotations'] if entry['image_id'] == image_id]
    single_image = [entry for entry in data['images'] if entry['id'] == image_id]

    plt.figure()
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    # Display image
    filename_str = single_image[0]['file_name']
    img_str = re.findall(r'\d+' + r'.jpg', filename_str)[0]
    plt.imshow(np.asarray(PIL_Image.open(str(images_dir_path) + '/' + img_str)))

    for p in range(len(single_annot_image_data)):
        # Extract and display annotated points
        x = single_annot_image_data[p]['keypoints_x']
        y = single_annot_image_data[p]['keypoints_y']
        vis = single_annot_image_data[p]['keypoints_vis']
        # Extract visible keypoints only for display
        x_dsply= [coord for ind, coord in enumerate(x) if vis[ind] == 1]
        y_dsply= [coord for ind, coord in enumerate(y) if vis[ind] == 1]
        plt.scatter(x_dsply, y_dsply, c='r')
        # Extract and display bounding boxes
        bbox = single_annot_image_data[p]['bbox']
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()


# EDA
num_per_cat = defaultdict(list)
train_id_list = [img['id'] for img in data['images']]
for img in train_id_list:
    print(img)
    try:
        cat = get_category_name(img)
        num = get_num_persons(img)
        num_per_cat[cat].append(num)
    except IndexError as e:
        print(e)

avg_num_per_cat = {}
for key, value in num_per_cat.items():
    avg_num_per_cat[key] = np.ceil(mean(value))

keys_names = list(avg_num_per_cat.keys())
values = [value for key, value in avg_num_per_cat.items()]

# Plot results
plt.figure(figsize=(14, 16), dpi=80)
plt.scatter(values, keys_names, s=100, alpha=0.6)
plt.grid(linestyle='--', alpha=1)
for x, y, tex in zip(values, keys_names, keys_names):
    t = plt.text(x, y, tex, horizontalalignment='center',
                 verticalalignment='center', fontdict={'color': 'white'})
plt.title('Average Number of persons in activity category', fontdict={'size': 20})
plt.xlabel('$Number$')
plt.show()

# Chi-2 Hypothesis Test
chi, p = chi2(np.reshape(np.array(values), (-1, 1)), keys_names)
