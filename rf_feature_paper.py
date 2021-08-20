#!/usr/bin/env python
# coding: utf-8

# In[65]:


from matplotlib import pyplot as plt
import numpy as np
import math

import pandas as pd

# https://www.usna.edu/Users/oceano/pguth/md_help/html/approx_equivalents.htm
UNIT_CELL_SIZE = 0.001      # 0.001° ~= 111 metres

df_rows = pd.read_csv("./taxi_dataset/validation_data.csv", sep='\n', header=None, nrows=200)
df_raw = df_rows[0].str.split(',', expand=True)
df_raw.head(10)


# In[66]:


df = df_raw.iloc[:,:]
# df.columns = ["start_time", "lon_1", "lat_1", "lon_2", "lat_2", "lon_3", "lat_3"]
df.head(10)


# In[67]:


def cal_dis(lat_1,lon_1,lat_2,lon_2):
    lon_1 = lon_1 * math.pi / 180
    lat_1 = lat_1 * math.pi / 180
    lon_2 = lon_2 * math.pi / 180
    lat_2 = lat_2 * math.pi / 180
    a = abs(lat_1 - lat_2)
    b = abs(lon_1 - lon_2)
    d = 2 * 6378.137 * np.arcsin(
        np.sqrt(np.sin(a / 2) * np.sin(a / 2) + np.cos(lat_1) * np.cos(lat_2) * np.sin(b / 2) * np.sin(b / 2)))
    return d


# In[68]:


from operator import attrgetter

class Trajectory:
    def __init__(self, id, df_row):
        self.id = id
        self.first_timestamp = int(df_row[0])
        self.radius_of_gyration = -1        # default value -1 since equation is sqrt, making -1 impossible
        self.entropy = -1                   # default value -1 since equation never returns -1
        
        # populate points array
        self.points = []
        self.points.append(Point(time=int(self.first_timestamp), lon=df_row[1], lat=df_row[2]))
        for i in range(3, df_row.size, 2):
            if df_row[i] is None or df_row[i]=='':
                break
            else:
                this_timestamp = int(self.first_timestamp) + (i//2)*15
                self.points.append(Point(time=this_timestamp, lon=df_row[i], lat=df_row[i+1]))

        # find trajectory min and max lat, lon
        self.min_lon = (min(self.points,key=attrgetter('lon')).lon)
        self.max_lon = (max(self.points,key=attrgetter('lon')).lon)
        self.min_lat = (min(self.points,key=attrgetter('lat')).lat)
        self.max_lat = (max(self.points,key=attrgetter('lat')).lat)        
        
                

    def get_points_info(self):
        i = 1
        for point in self.points:
            print(i, point)
            i+=1

    def calc_radius_of_gyration(self):
        point_center_lat = np.mean([point.lat for point in self.points])
        point_center_lon = np.mean([point.lon for point in self.points])

        temp_sum_rog = 0
        for point in self.points:
            temp_sum_rog += cal_dis(lat_1=point.lat, lon_1=point.lon, lat_2=point_center_lat, lon_2=point_center_lon)

        m = len(self.points)
        self.radius_of_gyration = math.sqrt(temp_sum_rog / m)
        
    def calc_entropy(self):
        length = self.max_lon - self.min_lon
        width  = self.max_lat - self.min_lat
        
        length_size = math.ceil(length / UNIT_CELL_SIZE) + 1
        width_size  = math.ceil(width  / UNIT_CELL_SIZE) + 1

        count_grid = [ [0]*width_size for i in range(length_size)]
        for point in self.points:
            x = round((point.lon - self.min_lon) / UNIT_CELL_SIZE)
            y = round((point.lat - self.min_lat) / UNIT_CELL_SIZE)
            try:
                count_grid[x][y] += 1
            except:
                print("traj id:", self.id)
                print("length_size:{}, width_size:{}".format(self.max_lon-self.min_lon,self.max_lat-self.min_lat))
                print("count_grid dim: {}x{}".format(length_size,width_size))
                print(point)
                print("lon_diff:({}), lat_diff:({})".format(point.lon-self.min_lon, point.lat-self.min_lat))
                print("x:({}), y:({})".format(x,y))
                print("")
        
        # print(pd.DataFrame(count_grid))

        m = len(self.points)
        temp_ent_sum = 0
        for x in range(len(count_grid)):
            for y in range(len(count_grid[x])):
                cell_count = count_grid[x][y]
                if cell_count == 0:
                    # temp_ent_sum += 0
                    continue
                else:
                    percent_i = cell_count / m                              # p(i)
                    temp_ent_sum += (percent_i) * math.log2(percent_i)      # summation
        
        self.entropy = -1 * temp_ent_sum
        

    def __str__(self):
        return("\n{:10s}: [ID: {:5d}, Time_First: {:10d}, Points: {}] \n{:10s}  [LAT_range: ({:5f}, {:5f}), LON_range: ({:5f}, {:5f}])".                    format("Trajectory", self.id, self.first_timestamp, len(self.points),                            "", self.min_lat, self.max_lat, self.min_lon, self.max_lon))
            
    def __repr__(self):
        # print("\nin __repr__, calling __str__")
        return str(self)


        
        

class Point:
    def __init__(self, time, lon, lat):
        self.timestamp = time
        self.lon = float(lon)
        self.lat = float(lat)
        
        self.is_truth = True        # True if point coordinate is truth value
        self.prev_pt_time = -1
        self.next_pt_time = -1

        # for rf target
        self.coor = [-1,-1]

    def set_truth_false(self):
        self.is_truth = False

    def set_prediction(self, pred_pt_lst):
        pred_time, pred_lon, pred_lat = pred_pt_lst
        if self.timestamp == pred_time:
            self.lon = pred_lon
            self.lat = pred_lat
        else:
            print("ERORR: time mismatch")
            print("actual t :", self.timestamp)
            print("predicted:", pred_time)
            print("\n")

    def __str__(self):
        return ("{:10s}: [Timestamp: {:10d}, Longitude: {:9f}, Latitude: {:9f}, Truth: {}]"            .format("Point", self.timestamp, self.lon, self.lat, self.is_truth))

    def __repr__(self):
        # print("\nin __repr__, calling __str__")
        return str(self)


# In[69]:


taxi_trajectories = []

counter = 0
for index, row in df.iterrows():
    taxi_trajectories.append(Trajectory(counter, row))
    taxi_trajectories[counter].calc_radius_of_gyration()
    taxi_trajectories[counter].calc_entropy()
    counter += 1

# Test output of first traj ROG, Ent
# taxi_trajectories[0].calc_radius_of_gyration()
# print("ROG:", taxi_trajectories[0].radius_of_gyration)
# taxi_trajectories[0].calc_entropy()
# print("Ent:", taxi_trajectories[0].entropy)


# In[70]:


global_min_lon = (min(taxi_trajectories, key=attrgetter('min_lon')).min_lon)
global_max_lon = (max(taxi_trajectories, key=attrgetter('max_lon')).max_lon)
global_min_lat = (min(taxi_trajectories, key=attrgetter('min_lat')).min_lat)
global_max_lat = (max(taxi_trajectories, key=attrgetter('max_lat')).max_lat)

print("global_min_lon:", global_min_lon); print("global_max_lon:", global_max_lon)
print("global_min_lat:", global_min_lat); print("global_max_lat:", global_max_lat)

grid_length = global_max_lon - global_min_lon
grid_width  = global_max_lat - global_min_lat

length_size = math.ceil(grid_length / UNIT_CELL_SIZE) + 1
width_size = math.ceil(grid_width / UNIT_CELL_SIZE) + 1
print("length:", length_size); print("width:" , width_size)


# In[71]:


point_grid = [ [[[]]]*width_size for i in range(length_size)]   # create a 2d array of lists of points (3D array of points)
count_grid = [ [0]*width_size for i in range(length_size)]

for traj in taxi_trajectories:
    for point in traj.points:
        x = round((point.lon - global_min_lon) / UNIT_CELL_SIZE)
        y = round((point.lat - global_min_lat) / UNIT_CELL_SIZE)

        point_grid[x][y].append(point)
        count_grid[x][y] += 1

        point.coor = [x,y]

        # print(point)
        # print("x:({}), y:({})".format(x,y))
        # print()

print(pd.DataFrame(count_grid))


# In[72]:


import random
def random_clear_total(sampling_rate, use_seed=0):
    # randomly clear points in whole grid_obj, except for first and last in each trajectory for linear implementation

    # append every point from grid_obj into ls_points, except first and last in traj
    ls_points = []
    for traj in taxi_trajectories:
        ls_points.extend(traj.points[1:len(traj.points)-1])

    num_total_points = len(ls_points)
    print("num total:", num_total_points)
    num_test_points = int(sampling_rate * num_total_points)
    print("num test:", num_test_points)

    if use_seed != 0:
        random.seed(use_seed)

    random_test_index = random.sample(range(num_total_points), num_test_points)
    # print(sorted(random_test_index))

    [ls_points[i].set_truth_false() for i in random_test_index]

random_clear_total(0.6, use_seed=1)


# In[73]:


# for traj in taxi_trajectories:
#     traj.get_points_info()


# In[74]:


features_ls = []

for traj in taxi_trajectories:
    traj_points = traj.points
    last_truth_pt = traj_points[0]
    next_truth_pt = traj_points[len(traj_points)-1]
    to_predict = False              # True when current pt is not truth value

    for curr_pt in traj_points:
        if to_predict is False:
            if curr_pt.is_truth is True:
                # all good, truth
                last_truth_pt = curr_pt
            else:
                # encountered point with non-truth value
                to_predict = True
                num_missing_data = 1
                features_ls.append([last_truth_pt.lon, last_truth_pt.lat, last_truth_pt.timestamp, curr_pt.timestamp, curr_pt.coor])
        
        else:
            # in a streak of non-truth points
            if curr_pt.is_truth is False:
                # curr_pt.prev_pt_time = last_truth_pt.timestamp
                num_missing_data += 1
                features_ls.append([last_truth_pt.lon, last_truth_pt.lat, last_truth_pt.timestamp, curr_pt.timestamp, curr_pt.coor])

            else:
                # found truth point
                to_predict = False
                next_truth_pt = curr_pt

                # print("ltt:", last_truth_pt.timestamp)
                # print("ntt:", next_truth_pt.timestamp)

                # [features_ls[i][-1:-1][next_truth_pt.lon, next_truth_pt.lat, next_truth_pt.timestamp, traj.radius_of_gyration, traj.entropy])\
                #     for i in range(len(features_ls)-1, len(features_ls)-num_missing_data-1, -1)]

                # print("nmd:", num_missing_data)
                curr_len = len(features_ls)
                for i in range(curr_len-1, curr_len-num_missing_data-1, -1):
                    # print(features_ls[i])
                    features_ls[i][-1:-1] = [next_truth_pt.lon, next_truth_pt.lat, next_truth_pt.timestamp, traj.radius_of_gyration, traj.entropy]
                    # print(features_ls[i])
                    # print()

                last_truth_pt = curr_pt


# In[75]:


# for i in range(len(features_ls)):
#     print(features_ls[i])

df_train = pd.DataFrame(features_ls)
df_train.columns = ["prev lon", "prev lat", "prev t", "curr t", "next lon", "next lat", "next t", "ROG", "Ent", "target"]
df_train


# In[76]:


# object type:  https://stackoverflow.com/questions/45346550/valueerror-unknown-label-type-unknown
# np.vstack:    https://stackoverflow.com/questions/19459017/how-to-convert-a-numpy-2d-array-with-object-dtype-to-a-regular-2d-array-of-float
# iloc:         https://stackoverflow.com/questions/55291667/getting-typeerror-slicenone-none-none-0-is-an-invalid-key
y = np.vstack(df_train.iloc[:,-1]).astype('int')
y


# In[77]:


from sklearn.model_selection import train_test_split

# y_test is truth target
X_train, X_test, y_train, y_test = train_test_split(df_train.drop(['target'],axis='columns'), y, test_size=0.2)

# X_train is the training data, X_test is testing data
print(len(X_train))
print(len(X_test))


# In[78]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

model.fit(X_train, y_train)
# model.get_params()


# In[79]:


y_predicted = model.predict(X_test)
print("y_predicted")
print(y_predicted[:20])

print("\ntruth:")
print(y_test[:20])


# In[80]:


# https://stackoverflow.com/questions/49830562/how-to-count-the-total-number-of-similar-elements-in-two-lists-occuring-at-the-s
# num_exact_match = sum(list(pred) == list(truth) for pred, truth in zip(y_predicted, y_test))
exact_matches = [i for i, (a, b) in enumerate(zip(y_predicted, y_test)) if list(a) == list(b)]
num_exact_match = len(exact_matches)
accuracy = num_exact_match/len(y_test)

print("Number of exact cell matches:", num_exact_match)
print("Accuracy:", accuracy)

# MAE
total_error_km = 0
error_count = 0
for i in range(len(y_predicted)):
    if i in exact_matches:
        continue

    error_count += 1
    grid_dist = math.dist(y_predicted[i], y_test[i])
    total_error_km += grid_dist * UNIT_CELL_SIZE * 111          # 1 degree = 111 km

mae = total_error_km / error_count
print("MAE in km:", mae)

