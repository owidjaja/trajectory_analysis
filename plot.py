from matplotlib import pyplot as plt
import numpy as np
import math

MIN_LAT = 41.14478; MIN_LON = -8.69346
MAX_LAT = 41.18652; MAX_LON = -8.57804

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

lenght = cal_dis(lat_1=MIN_LAT, lon_1=(MAX_LON-MIN_LON)/2, lat_2=MAX_LAT, lon_2=(MAX_LON-MIN_LON)/2)
width  = cal_dis(lat_1=(MAX_LAT-MIN_LAT)/2, lon_1=MIN_LON, lat_2=(MAX_LAT-MIN_LAT)/2, lon_2=MAX_LON)

print(lenght)
print(width)
print("area:", lenght*width)
