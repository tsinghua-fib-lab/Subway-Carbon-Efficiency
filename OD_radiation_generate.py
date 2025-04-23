from osgeo import gdal
import json
from tqdm import tqdm
import geopandas as gpd
from shapely import Polygon, Point, MultiPolygon
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from scipy.sparse import csr_matrix
from scipy.sparse import load_npz, save_npz
import heapq
import multiprocessing
import os
import argparse

parser = argparse.ArgumentParser(description='Subway')
parser.add_argument('--city_en', type=str, default='Wuxi')
parser.add_argument('--city_chn', type=str, default='无锡')
parser.add_argument('--city_en_short', type=str, default='Wuxi')
args = parser.parse_args()

city_en = args.city_en
city_en_short = args.city_en_short
city_chn = args.city_chn

ulx, uly, interval=(73.557916387, 53.561249988, 0.016666666666666666)

chn_shps = gpd.read_file('chn_boundary_2020_shp/chn_admbnda_adm2_ocha_2020.shp')
if city_en in ['Suzhou', 'Fuzhou', 'Taizhou']:
    city_boundary = chn_shps[chn_shps['ADM2_ZH']=='{}市'.format(city_chn)]['geometry'].values[0]
elif city_en == 'Chongqing':
    city_boundary = gpd.read_file('chn_boundary_2020_shp/Chongqing.shp')['geometry'][0]
elif city_en == 'Harbin':
    city_boundary = gpd.read_file('chn_boundary_2020_shp/Harbin.shp')['geometry'][0]
else:
    city_boundary = chn_shps[chn_shps['ADM2_EN'] == city_en]
    if len(city_boundary) > 1:
        print('Warning!! Repetive City Name!!', city_en)
        exit()
    city_boundary = city_boundary['geometry'].values[0]

min_lon, min_lat, max_lon, max_lat = city_boundary.bounds

year2popu = {}

def loc_cal(x, y):
    return ulx + (x + 0.5) * interval, uly - (y + 0.5) * interval

def grid_cal(lon, lat):
    return int((lon - ulx) / interval), int((uly - lat) / interval)

for year in tqdm(range(2010, 2021)):
    if year > 2020:
        continue
    file_path = "/data2/zengjinwei/subway_carbon/Population/chn_ppp_{}_UNadj.tif".format(year)
    dataset = gdal.Open(file_path)

    geo_information = dataset.GetGeoTransform()
    col_num = dataset.RasterXSize  # 73459
    row_num = dataset.RasterYSize  # 45538
    band = dataset.RasterCount  # 1
    dem_origin = dataset.GetRasterBand(1).ReadAsArray()

    block_num = 20
    dem = []
    for row in tqdm(range(0, row_num, block_num)):
        tmp = []
        for col in range(0, col_num, block_num):
            a = dem_origin[row:min(row + block_num, row_num), col:min(col + block_num, col_num)]
            a = a[a >= 0]
            tmp.append(sum(a))
        dem.append(tmp)

    # city boundary 
    xmin, ymin, xmax, ymax = city_boundary.bounds
    xmin_range, xmax_range = int(np.floor((xmin - ulx) / interval)), int(np.ceil((xmax - ulx) / interval))
    ymin_range, ymax_range = int(np.floor((uly - ymax) / interval)), int(np.ceil((uly - ymin) / interval))

    city_popu = {}
    for x in tqdm(range(xmin_range, xmax_range)):
        for y in range(ymin_range, ymax_range):
            lon, lat = loc_cal(x, y)
            city_popu[(x, y)] = dem[y][x]
    year2popu[year] = city_popu

year2popu_str = {}
for year in year2popu:
    year2popu_str[year] = dict(zip([str(m) for m in year2popu[year].keys()], year2popu[year].values()))

with open('city_popu_json/year_popu_{}.json'.format(city_en), 'w') as f:
    json.dump(year2popu_str, f)

### OD calculation
if os.path.exists('OD_{}'.format(city_en_short)) == False:
    os.mkdir('OD_{}'.format(city_en_short))

with open ('city_popu_json/year_popu_{}.json'.format(city_en), 'r') as f:
    year2popu = json.load(f)

grids_all_final = []
for key in year2popu['2020']:
    lon, lat = loc_cal(eval(key)[0], eval(key)[1])
    if city_boundary.contains(Point(lon, lat)):
        grids_all_final.append(eval(key))

np.save('OD_{}/grids_all_final.npy'.format(city_en_short), grids_all_final)

grids_all_final = np.load('OD_{}/grids_all_final.npy'.format(city_en_short), allow_pickle=True)
grids_all_final = [tuple(m.tolist()) for m in grids_all_final]

xmin, ymin, xmax, ymax = city_boundary.bounds
xmin_range, xmax_range = int(np.floor((xmin - ulx) / interval)), int(np.ceil((xmax - ulx) / interval))
ymin_range, ymax_range = int(np.floor((uly - ymax) / interval)), int(np.ceil((uly - ymin) / interval))

def grid2bound(grid):
    x, y = grid[0], grid[1]
    return ((ulx + x * interval, uly - y * interval), (ulx + (x+1) * interval, uly - y * interval), (ulx + (x+1) * interval, uly - (y+1) * interval), (ulx + x * interval, uly - (y+1) * interval))

def distance_cal(dlon, dlat, lat):
    R = 6371
    dlat = math.radians(dlat)
    dlon = math.radians(dlon)

    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat)) * math.cos(math.radians(lat + dlat)) * math.sin(
        dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

idx2distance = {}
lat = (max_lat + min_lat) / 2

distance_matrix = np.zeros((len(grids_all_final), len(grids_all_final)))
for i in tqdm(range(len(grids_all_final))):
    for j in range(len(grids_all_final)):
        grid0, grid1 = grids_all_final[i], grids_all_final[j]
        distance = (abs(grid0[0] - grid1[0]), abs(grid0[1] - grid1[1]))
        if distance in idx2distance:
            distance_matrix[i, j] = idx2distance[distance]
        else:
            tmp = distance_cal(distance[0] * interval, distance[1] * interval, lat)
            distance_matrix[i, j] = tmp
            idx2distance[distance] = tmp

for year in range(2010, 2021):
    city2popu = year2popu[str(year)]
    keys = [eval(m) for m in list(city2popu.keys())]
    city_popu = dict(zip(keys, list(city2popu.values())))

    ### UO model
    M = sum(city_popu.values())
    alpha = 0.08
    beta = 0.04
    
    city_popu_list = np.array([city_popu[grid] if grid in city_popu else 0 for grid in grids_all_final])
    
    def UO_cal(grid_id):
        distance = distance_matrix[grid_id, :]
        mi = city_popu[grids_all_final[grid_id]]
    
        distance2popu = {}
        for i in range(len(grids_all_final)):
            if grids_all_final[i] not in city_popu or city_popu[grids_all_final[i]] == 0:
                continue
            if distance[i] in distance2popu:
                distance2popu[distance[i]] += city_popu[grids_all_final[i]]
            else:
                distance2popu[distance[i]] = city_popu[grids_all_final[i]]
    
        origin_popu_list = list(distance2popu.values())
        distance_sorted = heapq.nsmallest(len(distance2popu), range(len(distance2popu)),
                                          np.array(list(distance2popu.keys())).take)
        
        popu_list = np.array(origin_popu_list)[distance_sorted][1:] 
        S_list = np.array([0] + list(np.cumsum(popu_list)[:-1])) 
        S_list = (mi + alpha * S_list) * popu_list / (mi + (alpha + beta) * S_list) / (
                    mi + (alpha + beta) * S_list + popu_list)
        S_list = np.array([S_list[m] / sum(S_list) for m in range(len(popu_list))]) * mi
    
        k = np.zeros((1, len(grids_all_final)))
        distance_sorted_list = sorted(list(distance2popu.keys()))[1:]
        for distance_idx in range(len(distance_sorted_list)):
            distance_tmp = distance_sorted_list[distance_idx]
            idxs = list(np.where(distance == distance_tmp)[0])
            flow = city_popu_list[idxs]
            flow = flow / np.sum(flow) * S_list[distance_idx]
            k[0, idxs] = flow
    
        return k, grid_id
    
    od_matrix_UO = np.zeros((len(grids_all_final), len(grids_all_final)))
    
    num_processes = 30
    pool = multiprocessing.Pool(processes=num_processes)
    
    args_list = []
    for i in range(len(grids_all_final)):
        if grids_all_final[i] in city_popu and city_popu[grids_all_final[i]]>0:
            args_list.append(i)
    
    with tqdm(total=len(args_list)) as pbar:
        for result in pool.imap(UO_cal, args_list):
            if result is not None:
                od_matrix_UO[result[-1], :] = result[0]
            pbar.update()
    
    pool.close()
    pool.join()
    
    row, col = np.nonzero(od_matrix_UO)
    values = od_matrix_UO[row, col].tolist()
    grid_od_csr = csr_matrix((values, (row, col)), shape=od_matrix_UO.shape)
    save_npz('OD_{}/UO_od_WorldPop_{}.npz'.format(city_en_short, year), grid_od_csr, compressed=True)
    print('UO with WorldPop finished!')

