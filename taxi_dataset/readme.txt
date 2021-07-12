1.  The taxi dataset was collected by all the 422 taxis running in the city of Porto, in Portugal over 12 months. These taxis operate through a taxi dispatch center, using mobile data terminals installed in the
vehicles to collect the location data. Each taxi reports its location every 15 seconds. 

2. The data are sampled from the following range:
max_lat = 41.18652
min_lat = 41.14478
max_lon = -8.57804
min_lon = -8.69346

3. data formate:

start_time, lon_1, lat_1, lon_2, lat_2, lon_3, lat_3, ..., lon_n, lat_n


4. An example to calculate the distance between locations:

import math
import numpy as np
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





