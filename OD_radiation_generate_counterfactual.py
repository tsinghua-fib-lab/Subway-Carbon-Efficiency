import requests
import random
from osgeo import gdal
import json
from tqdm import tqdm
import geopandas as gpd
from shapely import Polygon, Point, MultiPolygon
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, load_npz, save_npz
import heapq
import multiprocessing
import os
import logging
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
import argparse
import copy
import datetime

today = datetime.date.today()
now = datetime.datetime.now()

parser = argparse.ArgumentParser(description='Subway')
parser.add_argument('--city_en', type=str, default='Wuxi')
parser.add_argument('--city_chn', type=str, default='无锡')
parser.add_argument('--city_en_short', type=str, default='Wuxi')
args = parser.parse_args()

city_en = args.city_en
city_en_short = args.city_en_short
city_chn = args.city_chn

print('City: {}'.format(city_en_short))

basic_dir = 'route_api_hours/{}'.format(city_en_short)
if os.path.exists(basic_dir) == False:
    os.mkdir(basic_dir)

logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename=basic_dir+'/counterfactual_od_generation_{}_{}.log'.format(city_en_short, today),  # 将日志记录到文件
                    filemode='w')  
logging.info("City: {}".format(city_en_short))
logging.info('Time: {}'.format(now))

def loc_cal(x, y):
    return ulx + (x+0.5) * interval, uly - (y+0.5) * interval

def grid_cal(lon, lat):
    return int((lon - ulx) / interval), int((uly - lat) / interval)

ulx, uly, interval=(73.557916387, 53.561249988, 0.016666666666666666)

chn_shps = gpd.read_file('chn_boundary_2020_shp/chn_admbnda_adm2_ocha_2020.shp')
if city_en in ['Suzhou', 'Fuzhou', 'Taizhou']:
    city_boundary = chn_shps[chn_shps['ADM2_ZH']=='{}市'.format(city_chn)]['geometry'].values[0]
elif city_en == 'Chongqing':
    city_boundary = gpd.read_file('chn_boundary_2020_shp/Chongqing.shp')['geometry'][0]
elif city_en == 'Harbin':
    city_boundary = gpd.read_file('chn_boundary_2020_shp/Harbin.shp')['geometry'][0]
else:
    city_boundary = chn_shps[chn_shps['ADM2_EN'] == city_en]['geometry'].values[0]

min_lon, min_lat, max_lon, max_lat = city_boundary.bounds

subway_years = pd.read_excel('subway_boundary_file/subway_built_year/subway_years_{}.xlsx'.format(city_en_short), dtype={'connect': str})
for row in range(len(subway_years)):
    if type(subway_years.iloc[row]['地铁线路']) == str:
        name = subway_years.iloc[row]['地铁线路']
    else:
        subway_years.loc[row, '地铁线路'] = name
subway_stations_years = pd.read_excel('subway_boundary_file/subway_built_year/subway_stations_years_{}.xlsx'.format(city_en_short), sheet_name=None)

if os.path.exists('subway_boundary_file/station2loc/{}.json'.format(city_en_short)):
    with open('subway_boundary_file/station2loc/{}.json'.format(city_en_short), 'r') as f:
        station2loc = json.load(f)
else:
    with open(basic_dir+'/given_od_filtered_subway_{}_combined.json'.format(city_en), 'r') as f:
        subway_json = json.load(f)

    station2loc = {}
    for key in subway_json:
        if len(subway_json[key]['route']['transits']) > 0:
            for transit in subway_json[key]['route']['transits']:
                for m in transit['segments']:
                    if 'bus' in m:
                        for busline in m['bus']['buslines']:
                            if busline['type'] == '地铁线路':
                                if busline['departure_stop']['name'] not in station2loc:
                                    station2loc[busline['departure_stop']['name']] = eval(busline['departure_stop']['location'])
                                if busline['arrival_stop']['name'] not in station2loc:
                                    station2loc[busline['arrival_stop']['name']] = eval(busline['arrival_stop']['location'])
    station_set = set([])
    for line in subway_stations_years:
        stations = set(subway_stations_years[line]['站点'].tolist())
        station_set = station_set.union(stations)

    print('Missing stations:')
    missing_stations = []
    for station in station_set:
        if station not in station2loc:
            missing_stations.append(station)
            print(station)
    logging.info('Missing stations: {}'.format(missing_stations))

    for station in missing_stations:
        params = {
            'key': '0aade5f89420556935d8cace00dd3fde',
            'keywords': station+'站',    
            'city': city_chn}
        result = requests.get('https://restapi.amap.com/v3/place/text?parameters', params).json()['pois']
        if len(result) > 0:
            loc = result[0]['location']
            station2loc[station] = eval(loc)

    with open('subway_boundary_file/station2loc/{}.json'.format(city_en_short), 'w') as f:
        json.dump(station2loc, f)

subway_years['open_time'] = subway_years['开通时间'].apply(lambda x: x.split(' ')[1] if len(x.split(' ')) > 1 else x)
subway_years['open_month'] = subway_years['open_time'].apply(lambda x: int(x.split('.')[1]))
subway_years['open_year'] = subway_years['open_time'].apply(lambda x: int(x.split('.')[0]))

def duan_oneway_update(subway_duans, new_duan, connect):
    for duan in subway_duans:
        if connect in duan:
            subway_duans = subway_duans - set([duan])
            if duan[0] == connect:
                new_duan = (new_duan['begin'], duan[1])
            else:
                new_duan = (duan[0], new_duan['end'])
            subway_duans = subway_duans | set([new_duan])
    return subway_duans

def duan_twoway_update(subway_duans, new_duan, connects):
    for connect in connects:
        for duan in subway_duans:
            if connect in duan:
                subway_duans = subway_duans - set([duan])
                if duan[0] == connect:
                    dest = duan[1]
                else:
                    origin = duan[0]
    new_duan = (origin, dest)
    subway_duans = subway_duans | set([new_duan])
    return subway_duans


def subway_duan_update(subway_duans, new_duan):
    if type(new_duan['connect']) != float:
        try:
            connects = eval(new_duan['connect'])
            subway_duans = duan_twoway_update(subway_duans, new_duan, connects)
        except:
            if type(new_duan['connect']) != float and type(new_duan['connect']) != np.float64:
                for duan in subway_duans:
                    if new_duan['connect'] in duan:
                        subway_duans = duan_oneway_update(subway_duans, new_duan, new_duan['connect'])
                        break
    else:
        subway_duans = subway_duans | set([(new_duan['begin'], new_duan['end'])])
    return subway_duans

year2line_operation = {}
for year in range(2009, 2024):
    year2line_operation[year] = {}
    new_built = subway_years[subway_years['open_year']==year]
    line_operation = {}
    for line in subway_stations_years:
        signal = 1
        new_duans = subway_years[(subway_years['地铁线路']==line)&(subway_years['open_year']<year)]
        if len(new_duans) == 0:
            continue
        subway_duans = set([])
        for idx in range(len(new_duans)):
            new_duan = new_duans.iloc[idx]
            subway_duans = subway_duan_update(subway_duans, new_duan)
            if subway_duans is not None:
                line_operation[line] = subway_duans
    if len(new_built) == 0:
        if len(line_operation) != 0:
            year2line_operation[year][tuple(list(range(1, 13)))] = copy.deepcopy(line_operation)
    else:
        months = subway_years[subway_years['open_year'] == year]['open_month'].unique().tolist()
        months = sorted([int(m) for m in months])
        if len(line_operation) != 0:
            year2line_operation[year][tuple(list(range(1, int(months[0]) + 1)))] = copy.deepcopy(line_operation)
        for month_idx in range(len(months)):
            month = months[month_idx]
            if month == 12:
                continue
            next_month = int(months[month_idx + 1]) if month_idx + 1 < len(months) else 12
            for line in subway_stations_years:
                newduans = pd.concat([subway_years[(subway_years['地铁线路'] == line) &(subway_years['open_year']<year)], subway_years[(subway_years['地铁线路'] == line) &(subway_years['open_year']==year)&(subway_years['open_month']<=month)]], axis=0)
                if len(newduans) == 0:
                    continue
                subway_duans = set([])
                for idx in range(len(newduans)):
                    new_duan = newduans.iloc[idx]
                    subway_duans = subway_duan_update(subway_duans, new_duan)
                    if subway_duans is not None:
                        line_operation[line] = subway_duans
            if len(line_operation) != 0:
                year2line_operation[year][tuple(list(range(month + 1, next_month + 1)))] = copy.deepcopy(line_operation)

logging.info("year2line_operation is generated: %s" % str(year2line_operation))

year2station_operation = {}
for year in range(2009, 2024):
    year2station_operation[year] = {}
    for months in year2line_operation[year]:
        year2station_operation[year][str(months)] = {}
        for line in year2line_operation[year][months]:
            subway_stations_duans = []
            line_stations = np.array(list(subway_stations_years[line]['站点']))
            for duan in year2line_operation[year][months][line]:
                origin = np.where(line_stations==duan[0])[0][0]
                dest = np.where(line_stations==duan[1])[0][0]
                if origin < dest:
                    station_duans = set(line_stations[origin:dest+1])
                else:
                    station_duans = set(line_stations[dest:origin+1])
                subway_stations_duans.append(station_duans)
            year2station_operation[year][str(months)][line] = subway_stations_duans

for year in year2station_operation:
    for months in year2station_operation[year]:
        for line in year2station_operation[year][months]:
            year2station_operation[year][months][line] = str(year2station_operation[year][months][line])

with open('subway_boundary_file/subway_built_year/fenduan_{}.json'.format(city_en), 'w') as f:
    json.dump(year2station_operation, f)

with open('/data2/zengjinwei/subway_carbon/subway_boundary_file/subway_built_year/fenduan_{}.json'.format(city_en), 'r') as f:
    fenduan_info = json.load(f)
subway_years = pd.read_excel('subway_boundary_file/subway_built_year/subway_years_{}.xlsx'.format(city_en_short), dtype={'connect': str})
for row in range(len(subway_years)):
    if type(subway_years.iloc[row]['地铁线路']) == str:
        name = subway_years.iloc[row]['地铁线路']
    else:
        subway_years.loc[row, '地铁线路'] = name

subway_years['open_time'] = subway_years['开通时间'].apply(lambda x: x.split(' ')[1] if len(x.split(' ')) > 1 else x)
subway_years['open_month'] = subway_years['open_time'].apply(lambda x: x.split('.')[1])
subway_years['open_year'] = subway_years['open_time'].apply(lambda x: x.split('.')[0])
subway_years = subway_years[subway_years['remove'] != 1]    ### 考虑研究范围
lines = subway_years['地铁线路'].unique()

year2station = {}
for year in range(2009, 2024):
    year2station[year] = []
    for line in lines:
        min_year = min(subway_years[subway_years['地铁线路']==line]['open_year'])
        min_month = min(subway_years[(subway_years['地铁线路']==line)&(subway_years['open_year']==min_year)]['open_month'])
        if min_year > str(year) or (min_year == str(year) and min_month == '12'):
            continue
        if len(subway_years[subway_years['地铁线路']==line]) == len(subway_years[(subway_years['地铁线路']==line) & (subway_years['open_year']<= str(year))]):
            year2station[year] += subway_stations_years[line]['站点'].unique().tolist()
        else:
            for months in fenduan_info[str(year)]:
                if 12 in eval(months):
                    year2station[year] += list(eval(fenduan_info[str(year)][months][line])[0])

grid2yearstation = {}

for year in year2station:
    grid2yearstation[year] = {}
    for station in year2station[year]:
        if station in station2loc:
            if grid_cal(station2loc[station][0], station2loc[station][1]) in grid2yearstation[year]:
                grid2yearstation[year][grid_cal(station2loc[station][0], station2loc[station][1])] += 1
            else:
                grid2yearstation[year][grid_cal(station2loc[station][0], station2loc[station][1])] = 1

### panel regression
with open ('city_popu_json/year_popu_{}.json'.format(city_en), 'r') as f:
    year2popu = json.load(f)
grid_list = []
year_list = []
popu_list = []
stationnum_list = []

for grid_str in year2popu['2010']:
    grid = eval(grid_str)
    for year in range(2010, 2021):
        grid_list.append(grid)
        year_list.append(year)
        popu_list.append(year2popu[str(year)][grid_str])
        if grid in grid2yearstation[year]:
            stationnum_list.append(grid2yearstation[year][grid])
        else:
            stationnum_list.append(0)

df = pd.DataFrame({'grid': grid_list, 'year': year_list, 'stationnum': stationnum_list, 'popu': popu_list, })
df = df.set_index(['grid', 'year'])

exog = sm.add_constant(df[['stationnum']])
model = PanelOLS(df['popu'], exog, entity_effects=True, time_effects=True)
model = model.fit()

grid_list = []
year_list = []
stationnum_list = []

for year in range(2010, 2021):
    for grid in grid2yearstation[year]:
        if grid not in year2popu['2010']:
            continue         
        if grid not in grid2yearstation[2009] or grid2yearstation[year][grid] > grid2yearstation[2009][grid]:
            stationnum = grid2yearstation[2009][grid] if grid in grid2yearstation[2009] else 0
            grid_list.append(grid)
            year_list.append(year)
            stationnum_list.append(stationnum)

df_counterfactual = pd.DataFrame({'grid': grid_list, 'year': year_list, 'stationnum': stationnum_list})
df_counterfactual = df_counterfactual.set_index(['grid', 'year'])
exog = sm.add_constant(df_counterfactual[['stationnum']])
predict = model.predict(exog)
entity_effect = model.estimated_effects

result_counterfactual = predict.reset_index().merge(entity_effect.reset_index(), on=['grid', 'year'], how='left')
result_counterfactual['counterfactual_predictions'] = result_counterfactual.apply(lambda row: row['predictions']+row['estimated_effects'], axis=1)
result_counterfactual = result_counterfactual.merge(df.reset_index(), on=['grid', 'year'], how='left')
result_counterfactual['counterfactual_predictions'] = result_counterfactual['counterfactual_predictions'].apply(lambda x: 0 if x < 0 else x)   

year2popu_counteractual = copy.deepcopy(year2popu)

for row_id in range(len(result_counterfactual)):
    row = result_counterfactual.iloc[row_id]
    year2popu_counteractual[str(row['year'])][str(row['grid'])] = row['counterfactual_predictions']

with open ('city_popu_json/year_popu_{}_counterfactual.json'.format(city_en), 'w') as f:
    json.dump(year2popu_counteractual, f)

### counterfactual OD generation
year2popu = year2popu_counteractual

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
    alpha = 0.08
    beta = 0.04

    city_popu_list = np.array([city_popu[grid] if grid in city_popu else 0 for grid in grids_all_final])
    M = sum(city_popu_list)

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

    num_processes = 50
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
    save_npz('OD_{}/UO_od_WorldPop_{}_counterfactual.npz'.format(city_en_short, year), grid_od_csr, compressed=True)
