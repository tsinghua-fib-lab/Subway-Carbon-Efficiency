import requests
import random
# from osgeo import gdal
import json
from tqdm import tqdm
import geopandas as gpd
from shapely import Polygon, Point, MultiPolygon
import numpy as np
import math
import pandas as pd
from scipy.sparse import load_npz, save_npz
import numpy as npxf
import logging
import datetime
import os
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import copy
import argparse
import time
from billiard.pool import Pool
from pypinyin import pinyin, Style

parser = argparse.ArgumentParser(description='Subway')
parser.add_argument('--city_en', type=str, default='Beijing Municipality')
parser.add_argument('--city_chn', type=str, default='北京')
parser.add_argument('--city_en_short', type=str, default='Beijing')
parser.add_argument('--year', type=int, default=2019)
parser.add_argument('--parking_fee', type=int, default=20)  
args = parser.parse_args()

today = datetime.date.today()

if len(args.city_en.split(' '))>1:
    args.num_process = 5
else:
    args.num_process = 8
print('num_process: ', args.num_process)

print('1. City Basic Setting')
ulx, uly, interval = (73.557916387, 53.561249988, 0.016666666666666666)
interval_new = interval * 3  

city_en = args.city_en
city_en_short = args.city_en_short
city_chn = args.city_chn

elec_table = pd.read_excel('real_statistics/electricity_consumption/Electricity_consumption.xlsx')
elec_intensity = elec_table[elec_table['City']==city_en_short]['per_mileage_carbon (tCO2/km)']

vehicle_occupancy = 1.26
commute_ratio = 1
noncommute_ratio = 0

api_dir = 'route_api_hours/{}'.format(city_en_short)
basic_dir = 'route_api_hours/20240625/{}/result_files'.format(city_en_short)
if os.path.exists('route_api_hours/20240625/{}'.format(city_en_short)) == False:
    os.mkdir('route_api_hours/20240625/{}'.format(city_en_short))
if os.path.exists(basic_dir) == False:
    os.mkdir(basic_dir)

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename=basic_dir + '/{}_{}_{}.log'.format(city_en_short, city_en_short, today),
                    filemode='w') 
logging.info("City: {}".format(city_en_short))
logging.info('Parking fee: {}'.format(args.parking_fee))

parking_fee_overall = args.parking_fee

chn_shps = gpd.read_file('chn_boundary_2020_shp/chn_admbnda_adm2_ocha_2020.shp')
if city_en in ['Suzhou']:
    city_boundary = chn_shps[chn_shps['ADM2_ZH'] == '{}市'.format(city_chn)]['geometry'].values[0]
else:
    city_boundary = chn_shps[chn_shps['ADM2_EN'] == city_en]
    city_boundary = city_boundary['geometry'].values[0]
city_lat = city_boundary.centroid.y

xmin, ymin, xmax, ymax = city_boundary.bounds
xmin_range, xmax_range = int(np.floor((xmin - ulx) / interval)), int(np.ceil((xmax - ulx) / interval))
ymin_range, ymax_range = int(np.floor((uly - ymax) / interval)), int(np.ceil((uly - ymin) / interval))

def loc_cal(x, y):
    return ulx + (x + 0.5) * interval, uly - (y + 0.5) * interval

def new_loc_cal(x, y):
    return ulx + (x + 0.5) * interval_new, uly - (y + 0.5) * interval_new

def grid_cal(lon, lat):
    return int((lon - ulx) / interval), int((uly - lat) / interval)

def new_grid_cal(lon, lat):
    return int((lon - ulx) / interval_new), int((uly - lat) / interval_new)

def grid_transform(grid):
    x, y = grid[0], grid[1]
    loc = loc_cal(x, y)
    new_grid = new_grid_cal(loc[0], loc[1])
    return new_grid

grids = np.load('OD_{}/grids_all_final.npy'.format(city_en_short))
large_grid_pairs = np.load('OD_{}/large_grid_pairs.npy'.format(city_en_short), allow_pickle=True)
large_grid_pairs = [(tuple(m[0]), tuple(m[1])) for m in large_grid_pairs]

### saved time and monetary cost files for subway, drive, bicycle, bus
with open(api_dir + '/given_od_filtered_subway_{}_combined.json'.format(city_en), 'r') as f:
    subway_json = json.load(f)
with open(api_dir + '/given_od_filtered_drive_{}_combined.json'.format(city_en), 'r') as f:
    drive_json = json.load(f)
with open(api_dir + '/given_od_filtered_bicycle_{}_combined.json'.format(city_en), 'r') as f:
    bicycle_json = json.load(f)
with open(api_dir + '/given_od_filtered_bus_{}_combined.json'.format(city_en), 'r') as f:
    bus_json = json.load(f)

print('2. Subway Basic Setting')
subway_years = pd.read_excel('subway_boundary_file/subway_built_year/subway_years_{}.xlsx'.format(city_en_short),
                             dtype={'connect': str})
subway_stations_years = pd.read_excel(
    'subway_boundary_file/subway_built_year/subway_stations_years_{}.xlsx'.format(city_en_short), sheet_name=None)

for row in range(len(subway_years)):
    if type(subway_years.iloc[row]['地铁线路']) == str:
        name = subway_years.iloc[row]['地铁线路']
    else:
        subway_years.loc[row, '地铁线路'] = name

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
    new_built = subway_years[subway_years['open_year'] == year]
    line_operation = {}
    for line in subway_stations_years:
        signal = 1
        new_duans = subway_years[(subway_years['地铁线路'] == line) & (subway_years['open_year'] < year)]
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
                newduans = pd.concat([subway_years[(subway_years['地铁线路'] == line) & (subway_years['open_year'] < year)],
                                      subway_years[
                                          (subway_years['地铁线路'] == line) & (subway_years['open_year'] == year) & (
                                                  subway_years['open_month'] <= month)]], axis=0)
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

year2station_operation = {}
for year in range(2009, 2024):
    year2station_operation[year] = {}
    for months in year2line_operation[year]:
        year2station_operation[year][str(months)] = {}
        for line in year2line_operation[year][months]:
            subway_stations_duans = []
            line_stations = np.array(list(subway_stations_years[line]['站点']))
            for duan in year2line_operation[year][months][line]:
                origin = np.where(line_stations == duan[0])[0][0]
                dest = np.where(line_stations == duan[1])[0][0]
                if origin < dest:
                    station_duans = set(line_stations[origin:dest + 1])
                else:
                    station_duans = set(line_stations[dest:origin + 1])
                subway_stations_duans.append(station_duans)
            year2station_operation[year][str(months)][line] = subway_stations_duans

for year in year2station_operation:
    for months in year2station_operation[year]:
        for line in year2station_operation[year][months]:
            year2station_operation[year][months][line] = str(year2station_operation[year][months][line])

with open('result_files/subway_auguration_{}.json'.format(city_en), 'w') as f:
    json.dump(year2station_operation, f)

with open('result_files/subway_auguration_{}.json'.format(city_en), 'r') as f:
    subway_info = json.load(f)

year = args.year
alpha = 0.08
beta = 0.04
### load calculated OD matrix
od_matrix = load_npz('OD_{}/UO_od_WorldPop_{}.npz'.format(city_en_short, min(year, 2020))).todense().astype(np.float32).tolist()

month = 6
for months in year2station_operation[year]:
    if month in eval(months):
        month_range = months
logging.info('Year: %s, Month: %s' % (year, month))

def subway_line_distribute_total(grid1, grid2):
    def subway_judge(transit):
        for m in transit['segments']:
            if 'bus' in m:
                for busline in m['bus']['buslines']:
                    if busline['type'] == '地铁线路':
                        return True
        return False

    def subway_dict_return(transit):
        subway_dict = set([])
        for m in transit['segments']:
            if 'bus' in m:
                for busline in m['bus']['buslines']:
                    if busline['type'] == '地铁线路':
                        subway_dict.add(busline['name'])
        return subway_dict

    subway_set = set([])
    if str((grid1, grid2)) in subway_json.keys():
        if len(subway_json[str((grid1, grid2))]['route']['transits']) > 0:
            for transit in subway_json[str((grid1, grid2))]['route']['transits']:
                if subway_judge(transit):
                    subway_dict = subway_dict_return(transit)
                    subway_set = subway_set.union(subway_dict)
            return subway_set
    if str((grid2, grid1)) in subway_json.keys():
        if len(subway_json[str((grid2, grid1))]['route']['transits']) > 0:
            for transit in subway_json[str((grid2, grid1))]['route']['transits']:
                if subway_judge(transit):
                    subway_dict = subway_dict_return(transit)
                    subway_set = subway_set.union(subway_dict)
            return subway_set
    return {}

subway_set = set([])
for i in tqdm(range(len(grids))):
    for j in range(i + 1, len(grids)):
        grid0, grid1 = list(grids[i]), list(grids[j])
        if od_matrix[i][j] >= 5 or od_matrix[j][i] >= 5:
            subway_od_line_set = subway_line_distribute_total(grid0, grid1)
            subway_set = subway_set.union(subway_od_line_set)
print('subway_set: ', subway_set)

hour_list = [2, 7, 8, 17, 18]

hour2subway, hour2drive, hour2bicycle, hour2bus = {}, {}, {}, {}
def subway_judge(transit):
    for m in transit['segments']:
        if 'bus' in m:
            for busline in m['bus']['buslines']:
                if busline['type'] == '地铁线路':
                    return True
    return False

def subway_walking_time_scale(transit):
    walking_time, bus_time = 0, 0
    for segment in transit['segments']:
        if 'duration' in segment['walking']:
            walking_time += float(segment['walking']['duration'])
        for busline in segment['bus']['buslines']:
            bus_time += float(busline['duration'])
    return walking_time / 2 + bus_time

def subway_history_judge(transit):
    for m in transit['segments']:
        if 'bus' in m:
            for busline in m['bus']['buslines']:
                if busline['type'] == '地铁线路':
                    if busline['name'] not in subway_filtered_set:
                        return False
                    signal = 0
                    name = busline['name'].split('线')[0] + '线'
                    if name not in subway_info[str(year)][month_range]:
                        signal = 1
                    else:
                        departure = busline['departure_stop']['name']
                        dest = busline['arrival_stop']['name']
                        for duan in eval(subway_info[str(year)][month_range][name]):
                            if (departure in duan) & (dest in duan):
                                signal = 1
                                break
                            if (departure in duan) ^ (dest in duan):
                                return False
                    if signal == 0:
                        return False
    return True

count = 0

for hour in tqdm(hour_list):
    drive_duration_dict, bicycle_duration_dict, bus_duration_dict = {}, {}, {}
    with open(api_dir + '/subway_{}_{}.json'.format(city_en, hour), 'r') as f:
        subway_json_hours = json.load(f)
    with open(api_dir + '/drive_{}_{}.json'.format(city_en, hour), 'r') as f:
        drive_json_hours = json.load(f)
    with open(api_dir + '/bicycle_{}_{}.json'.format(city_en, hour), 'r') as f:
        bicycle_json_hours = json.load(f)
    with open(api_dir + '/bus_{}_{}.json'.format(city_en, hour), 'r') as f:
        bus_json_hours = json.load(f)

    for key in subway_json_hours:
        drive_duration = float(drive_json_hours[key]['route']['paths'][0]['duration'])
        bicycle_duration = float((bicycle_json_hours[key]['data']['paths'][0]['duration']))
        drive_duration_dict[key] = drive_duration
        bicycle_duration_dict[key] = bicycle_duration

        bus_exist = 0
        for transit in subway_json_hours[key]['route']['transits']:
            if not subway_judge(transit):
                bus_duration = float(transit['duration'])
                bus_exist = 1
                bus_duration_dict[key] = bus_duration
                break
        if bus_exist == 0 and key not in bus_json_hours:
            count += 1
        if bus_exist == 0 and key in bus_json_hours:
            if len(bus_json_hours[key]['route']['transits']) > 0:
                bus_duration = float(bus_json_hours[key]['route']['transits'][0]['duration'])
                bus_duration_dict[key] = bus_duration

    hour2drive[hour] = drive_duration_dict
    hour2bicycle[hour] = bicycle_duration_dict
    hour2bus[hour] = bus_duration_dict

for hour in tqdm(hour_list):
    hour2subway[hour] = {}
    with open(api_dir + '/subway_{}_{}.json'.format(city_en, hour), 'r') as f:
        subway_json_hours = json.load(f)
    for year in range(2010, 2024):
        hour2subway[hour][year] = {}
        for months in subway_info[str(year)]:
            subway_duration_dict = {}
            month_range = months
            month = eval(month_range)[0]
            subway_filtered_set = set([])
            available_subway = list(subway_years[subway_years['open_year'] < year]['地铁线路']) + list(
                subway_years[(subway_years['open_year'] == year) & (subway_years['open_month'] < month)]['地铁线路'])
            for subway in subway_set:
                if subway.split('线')[0] + '线' in available_subway:
                    subway_filtered_set.add(subway)

            for key in subway_json_hours:
                if len(subway_json_hours[key]['route']['transits']) > 0:
                    for transit in subway_json_hours[key]['route']['transits']:
                        if subway_judge(transit) and subway_history_judge(transit):  # 历史符合
                            subway_duration = float(transit['duration'])
                            subway_duration_dict[key] = subway_duration
                            break
            hour2subway[hour][year][month_range] = subway_duration_dict

commute2subway, commute2drive, commute2bicycle, commute2bus = {}, {}, {}, {}
noncommute2subway, noncommute2drive, noncommute2bicycle, noncommute2bus = {}, {}, {}, {}

for key in hour2bus[2]:
    drive_sum, drive_count = 0, 0
    drive_sum += hour2drive[7][key] if key in hour2drive[7] else 0
    drive_count += 1 if key in hour2drive[7] else 0
    drive_sum += hour2drive[8][key] if key in hour2drive[8] else 0
    drive_count += 1 if key in hour2drive[8] else 0
    drive_sum += hour2drive[17][key] if key in hour2drive[17] else 0
    drive_count += 1 if key in hour2drive[17] else 0
    drive_sum += hour2drive[18][key] if key in hour2drive[18] else 0
    drive_count += 1 if key in hour2drive[18] else 0
    if drive_count > 0:
        commute2drive[key] = drive_sum / drive_count / hour2drive[2][key]

for key in hour2bus[2]:
    bicycle_sum, bicycle_count = 0, 0
    bicycle_sum += hour2bicycle[7][key] if key in hour2bicycle[7] else 0
    bicycle_count += 1 if key in hour2bicycle[7] else 0
    bicycle_sum += hour2bicycle[8][key] if key in hour2bicycle[8] else 0
    bicycle_count += 1 if key in hour2bicycle[8] else 0
    bicycle_sum += hour2bicycle[17][key] if key in hour2bicycle[17] else 0
    bicycle_count += 1 if key in hour2bicycle[17] else 0
    bicycle_sum += hour2bicycle[18][key] if key in hour2bicycle[18] else 0
    bicycle_count += 1 if key in hour2bicycle[18] else 0
    if bicycle_count > 0:
        commute2bicycle[key] = bicycle_sum / bicycle_count / hour2bicycle[2][key]

for key in hour2bus[2]:
    bus_sum, bus_count = 0, 0
    bus_sum += hour2bus[7][key] if key in hour2bus[7] else 0
    bus_count += 1 if key in hour2bus[7] else 0
    bus_sum += hour2bus[8][key] if key in hour2bus[8] else 0
    bus_count += 1 if key in hour2bus[8] else 0
    bus_sum += hour2bus[17][key] if key in hour2bus[17] else 0
    bus_count += 1 if key in hour2bus[17] else 0
    bus_sum += hour2bus[18][key] if key in hour2bus[18] else 0
    bus_count += 1 if key in hour2bus[18] else 0
    if bus_count > 0:
        commute2bus[key] = bus_sum / bus_count / hour2bus[2][key]

for year in range(2010, 2024):
    commute2subway[year] = {}
    for months in subway_info[str(year)]:
        commute2subway[year][months] = {}
        for key in hour2subway[2][year][months]:
            subway_sum, subway_count = 0, 0
            subway_sum += hour2subway[7][year][months][key] if key in hour2subway[7][year][months] else 0
            subway_count += 1 if key in hour2subway[7][year][months] else 0
            subway_sum += hour2subway[8][year][months][key] if key in hour2subway[8][year][months] else 0
            subway_count += 1 if key in hour2subway[8][year][months] else 0
            subway_sum += hour2subway[17][year][months][key] if key in hour2subway[17][year][months] else 0
            subway_count += 1 if key in hour2subway[17][year][months] else 0
            subway_sum += hour2subway[18][year][months][key] if key in hour2subway[18][year][months] else 0
            subway_count += 1 if key in hour2subway[18][year][months] else 0
            if subway_count > 0:
                commute2subway[year][months][key] = subway_sum / subway_count / hour2subway[2][year][months][key]

print(len(commute2drive), len(commute2bicycle), len(commute2bus))

avecommute2drive = np.mean(list(commute2drive.values()))
avecommute2bus = np.mean(list(commute2bus.values()))
avecommute2bicycle = np.mean(list(commute2bicycle.values()))
avecommute2subway = {}
for year in range(2010, 2024):
    avecommute2subway[year] = {}
    for months in subway_info[str(year)]:
        avecommute2subway[year][months] = np.mean(list(commute2subway[year][months].values()))
avenoncommute2subway = 0
avenoncommute2drive = 0
avenoncommute2bus = 0
avenoncommute2bicycle = 0

grid2large_grid = {}
for grid in grids:
    grid = tuple(list(grid))
    grid2large_grid[grid] = tuple(grid_transform(grid))

print('3. Subway Filter Setting')
subway_filtered_set = set([])
available_subway = list(subway_years[subway_years['open_year'] < year]['地铁线路']) + list(
    subway_years[(subway_years['open_year'] == year) & (subway_years['open_month'] < month)]['地铁线路'])
for subway in subway_set:
    if subway.split('线')[0] + '线' in available_subway:
        subway_filtered_set.add(subway)

def distance_cal(dlon, dlat, lat):
    R = 6371
    dlat = math.radians(dlat)
    dlon = math.radians(dlon)

    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat)) * math.cos(math.radians(lat + dlat)) * math.sin(
        dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

lat = city_lat

grids_all_final = grids
idx2distance = {}
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


def route(args):
    grid1, grid2, i, j = args
    grid1, grid2 = list(grid1), list(grid2)

    large_grid1 = grid2large_grid[tuple(grid1)]
    large_grid2 = grid2large_grid[tuple(grid2)]
    if (large_grid1, large_grid2) in large_grid_pairs:
        key = str((large_grid1, large_grid2))
    elif (large_grid2, large_grid1) in large_grid_pairs:
        key = str((large_grid2, large_grid1))
    else:
        key = None

    def subway_judge(transit):
        for m in transit['segments']:
            if 'bus' in m:
                for busline in m['bus']['buslines']:
                    if busline['type'] == '地铁线路':
                        return True
        return False

    def subway_history_judge(transit):
        for m in transit['segments']:
            if 'bus' in m:
                for busline in m['bus']['buslines']:
                    if busline['type'] == '地铁线路':
                        if busline['name'] not in subway_filtered_set:
                            return False
                        signal = 0
                        name = busline['name'].split('线')[0] + '线'
                        if name not in subway_info[str(year)][month_range]:
                            signal = 1
                        else:
                            departure = busline['departure_stop']['name']
                            dest = busline['arrival_stop']['name']
                            for duan in eval(subway_info[str(year)][month_range][name]):
                                if (departure in duan) & (dest in duan):
                                    signal = 1
                                    break
                                if (departure in duan) ^ (dest in duan):
                                    return False
                        if signal == 0:
                            return False
        return True

    def bus_judge(bus_json):
        for m in bus_json['route']['transits'][0]['segments']:
            if 'bus' in m:
                return True
        return False

    def subway_mileage_cal(transit):
        mileage = 0
        for m in transit['segments']:
            if 'bus' in m:
                for busline in m['bus']['buslines']:
                    if busline['type'] == '地铁线路':
                        mileage += float(busline['distance'])
        return mileage

    if str((grid1, grid2)) in subway_json.keys():
        subway_exist = 0
        if len(subway_json[str((grid1, grid2))]['route']['transits']) > 0:
            for transit in subway_json[str((grid1, grid2))]['route']['transits']:
                if subway_exist == 0 and subway_judge(transit):
                    if not subway_history_judge(transit):
                        continue
                    subway_exist = 1
                    subway_duration = float(transit['duration'])
                    if len(transit['cost']) == 0:
                        subway_expense = 7
                    else:
                        subway_expense = float(transit['cost'])
                    subway_mileage = subway_mileage_cal(transit)
        if not subway_exist:
            subway_distance, subway_duration, subway_expense, subway_mileage = -1, -1, -1, -1

        bus_exist = 0
        if len(subway_json[str((grid1, grid2))]['route']['transits']) > 0:
            for transit in subway_json[str((grid1, grid2))]['route']['transits']:
                if not subway_judge(transit) and bus_exist == 0:
                    for m in transit['segments']:
                        if 'bus' in m:
                            bus_duration = float(transit['duration'])
                            bus_exist = 1
                            if len(transit['cost']) == 0:
                                bus_expense = 5
                            else:
                                bus_expense = float(transit['cost'])
        else:
            bus_distance, bus_duration, bus_expense = -1, -1, -1
            bus_exist = 1

        if not bus_exist:
            if len(bus_json[str((grid1, grid2))]['route']['transits']) > 0 and bus_judge(bus_json[str((grid1, grid2))]):
                bus_duration = float(bus_json[str((grid1, grid2))]['route']['transits'][0]['duration'])
                if len(bus_json[str((grid1, grid2))]['route']['transits'][0]['cost']) == 0:
                    bus_expense = 5
                else:
                    bus_expense = float(bus_json[str((grid1, grid2))]['route']['transits'][0]['cost'])
            else:
                bus_distance, bus_duration, bus_expense = -1, -1, -1

        drive_distance = float(drive_json[str((grid1, grid2))]['route']['paths'][0]['distance'])
        drive_duration = float(drive_json[str((grid1, grid2))]['route']['paths'][0]['duration'])

        bicycle_duration = float(bicycle_json[str((grid1, grid2))]['data']['paths'][0]['duration'])
        bicycle_distance = float(bicycle_json[str((grid1, grid2))]['data']['paths'][0]['distance'])
        if key in commute2subway[year][month_range]:
            if subway_duration > 0:
                subway_duration = subway_duration * commute2subway[year][month_range][key]
        else:
            if subway_duration > 0:
                subway_duration = subway_duration * avecommute2subway[year][month_range]
        if key in commute2drive:
            drive_duration = drive_duration * commute2drive[key]
        else:
            drive_duration = drive_duration * avecommute2drive
        if key in commute2bicycle:
            bicycle_duration = bicycle_duration * commute2bicycle[key]
        else:
            bicycle_duration = bicycle_duration * avecommute2bicycle
        if key in commute2bus:
            bus_duration = bus_duration * commute2bus[key]
        else:
            bus_duration = bus_duration * avecommute2bus
        return subway_expense, bus_expense, drive_duration, subway_duration, bus_duration, bicycle_duration, bicycle_distance, drive_distance, subway_mileage, grid1, grid2, i, j
    if str((grid2, grid1)) in subway_json.keys():
        subway_exist = 0
        if len(subway_json[str((grid2, grid1))]['route']['transits']) > 0:
            for transit in subway_json[str((grid2, grid1))]['route']['transits']:
                if subway_exist == 0 and subway_judge(transit):
                    if not subway_history_judge(transit):
                        continue
                    subway_exist = 1
                    subway_duration = float(transit['duration'])
                    if len(transit['cost']) == 0:
                        subway_expense = 7
                    else:
                        subway_expense = float(transit['cost'])
                    subway_mileage = subway_mileage_cal(transit)
        if not subway_exist:
            subway_distance, subway_duration, subway_expense, subway_mileage = -1, -1, -1, -1

        bus_exist = 0
        if len(subway_json[str((grid2, grid1))]['route']['transits']) > 0:
            for transit in subway_json[str((grid2, grid1))]['route']['transits']:
                if bus_exist == 0 and not subway_judge(transit):
                    for m in transit['segments']:
                        if 'bus' in m:
                            bus_duration = float(transit['duration'])
                            bus_exist = 1
                            if len(transit['cost']) == 0:
                                bus_expense = 5
                            else:
                                bus_expense = float(transit['cost'])
        else:
            bus_distance, bus_duration, bus_expense = -1, -1, -1
            bus_exist = 1

        if not bus_exist:
            if len(bus_json[str((grid2, grid1))]['route']['transits']) > 0 and bus_judge(bus_json[str((grid2, grid1))]):
                bus_duration = float(bus_json[str((grid2, grid1))]['route']['transits'][0]['duration'])
                if len(bus_json[str((grid2, grid1))]['route']['transits'][0]['cost']) == 0:
                    bus_expense = 5
                else:
                    bus_expense = float(bus_json[str((grid2, grid1))]['route']['transits'][0]['cost'])
            else:
                bus_distance, bus_duration, bus_expense = -1, -1, -1

        drive_distance = float(drive_json[str((grid2, grid1))]['route']['paths'][0]['distance'])
        drive_duration = float(drive_json[str((grid2, grid1))]['route']['paths'][0]['duration'])

        bicycle_duration = float(bicycle_json[str((grid2, grid1))]['data']['paths'][0]['duration'])
        bicycle_distance = float(bicycle_json[str((grid2, grid1))]['data']['paths'][0]['distance'])
        if key in commute2subway[year][month_range]:
            if subway_duration > 0:
                subway_duration = subway_duration * commute2subway[year][month_range][key]
        else:
            if subway_duration > 0:
                subway_duration = subway_duration * avecommute2subway[year][month_range]
        if key in commute2drive:
            drive_duration = drive_duration * commute2drive[key]
        else:
            drive_duration = drive_duration * avecommute2drive
        if key in commute2bicycle:
            bicycle_duration = bicycle_duration * commute2bicycle[key]
        else:
            bicycle_duration = bicycle_duration * avecommute2bicycle
        if key in commute2bus:
            bus_duration = bus_duration * commute2bus[key]
        else:
            bus_duration = bus_duration * avecommute2bus
        return subway_expense, bus_expense, drive_duration, subway_duration, bus_duration, bicycle_duration, bicycle_distance, drive_distance, subway_mileage, grid1, grid2, i, j

def subway_line_distribute(grid1, grid2, year, month_range, subway_filtered_set):
    grid1, grid2 = list(grid1), list(grid2)

    def subway_judge(transit):
        for m in transit['segments']:
            if 'bus' in m:
                for busline in m['bus']['buslines']:
                    if busline['type'] == '地铁线路':
                        return True
        return False

    def subway_history_judge(transit):
        for m in transit['segments']:
            if 'bus' in m:
                for busline in m['bus']['buslines']:
                    if busline['type'] == '地铁线路':
                        if busline['name'] not in subway_filtered_set:
                            return False
                        signal = 0
                        name = busline['name'].split('线')[0] + '线'
                        if name not in subway_info[str(year)][month_range]:
                            signal = 1
                        else:
                            departure = busline['departure_stop']['name']
                            dest = busline['arrival_stop']['name']
                            for duan in eval(subway_info[str(year)][month_range][name]):
                                if (departure in duan) & (dest in duan):
                                    signal = 1
                                    break
                                if (departure in duan) ^ (dest in duan):
                                    return False
                        if signal == 0:
                            return False
        return True

    def subway_dict_return(transit):
        subway_dict = {}
        for m in transit['segments']:
            if 'bus' in m:
                for busline in m['bus']['buslines']:
                    if busline['type'] == '地铁线路':
                        subway_dict[busline['name']] = busline['distance'] if busline[
                                                                                  'name'] not in subway_dict else \
                            busline['distance'] + subway_dict[busline['name']]
        return subway_dict

    if str((grid1, grid2)) in subway_json.keys():
        if len(subway_json[str((grid1, grid2))]['route']['transits']) > 0:
            for transit in subway_json[str((grid1, grid2))]['route']['transits']:
                if subway_judge(transit):
                    if not subway_history_judge(transit):
                        continue
                    subway_dict = subway_dict_return(transit)
                    return subway_dict
    if str((grid2, grid1)) in subway_json.keys():
        if len(subway_json[str((grid2, grid1))]['route']['transits']) > 0:
            for transit in subway_json[str((grid2, grid1))]['route']['transits']:
                if subway_judge(transit):
                    if not subway_history_judge(transit):
                        continue
                    subway_dict = subway_dict_return(transit)
                    return subway_dict
    return {}

def get_mode_distribution(args):
    subway_expense, bus_expense, driving_duration, subway_duration, bus_duration, bicycle_duration, distance, driving_distance, subway_mileage, od1, od2, grid0, grid1, year, month_range, subway_filtered_set = args
    if bicycle_duration == 0:
        return
    parking_fee = parking_fee_overall  
    age = 0.384  # average age
    income = 0.395  # average income
    if bus_expense > 0:
        V_bus = -0.0516 * bus_duration / 60 - 0.4810 * bus_expense
    else:
        V_bus = -np.inf
    if subway_expense > 0:
        V_subway = -0.0512 * subway_duration / 60 - 0.0833 * subway_expense
    else:
        V_subway = -np.inf
    V_fuel = -0.0705 * driving_duration / 60 + 0.5680 * age - 0.8233 * income - 0.0941 * parking_fee
    V_elec = -0.0339 * driving_duration / 60 - 0.1735 * parking_fee
    if distance > 15000:  # no cylcing for distance larger than 15km
        V_bicycle = -np.inf
    else:
        V_bicycle = -0.1185 * bicycle_duration / 60
    V = np.array([V_bus, V_subway, V_fuel, V_elec, V_bicycle])
    V = np.exp(V)
    V = V / sum(V)

    subway_line_distribution = subway_line_distribute(grid0, grid1, year, month_range, subway_filtered_set)
    return V[0], V[1], V[2], V[3], V[
        4], od1, od2, driving_distance, subway_mileage, grid0, grid1, subway_line_distribution

### construction carbon debt calculation
material = pd.read_excel('subway_construction_info/subway_construction_info.xlsx')
material2carbon = dict(zip(list(material.columns[1:]), material.iloc[0][1:]))

def construction_carbon_cal(station_num, mileage):
    entrance_exit_num, ventilization_num = int(station_num*4.3), int(station_num*4)
    carbon = material2carbon['Station_per_unit(tCO2e/unit)']*station_num + material2carbon['Tunnel_per_km(tCO2e/km)']/1e3*mileage + material2carbon['Entrance_per_unit(tCO2e/unit)']*entrance_exit_num + material2carbon['Ventilization_per_unit((tCO2e/unit))']*ventilization_num
    return carbon

print('4. Factual transportation distribution calculation')
def route(args):
    grid1, grid2, i, j, subway_filtered_set, year, month_range = args
    grid1, grid2 = list(grid1), list(grid2)

    large_grid1 = grid2large_grid[tuple(grid1)]
    large_grid2 = grid2large_grid[tuple(grid2)]
    key = (0, 0)
    if (large_grid1, large_grid2) in large_grid_pairs:
        key = str((large_grid1, large_grid2))
    if (large_grid2, large_grid1) in large_grid_pairs:
        key = str((large_grid2, large_grid1))

    def subway_judge(transit):
        for m in transit['segments']:
            if 'bus' in m:
                for busline in m['bus']['buslines']:
                    if busline['type'] == '地铁线路':
                        return True
        return False

    def subway_history_judge(transit):
        for m in transit['segments']:
            if 'bus' in m:
                for busline in m['bus']['buslines']:
                    if busline['type'] == '地铁线路':
                        if busline['name'] not in subway_filtered_set:
                            return False
                        signal = 0
                        name = busline['name'].split('线')[0] + '线'
                        if name not in subway_info[str(year)][month_range]:
                            signal = 1
                        else:
                            departure = busline['departure_stop']['name']
                            dest = busline['arrival_stop']['name']
                            for duan in eval(subway_info[str(year)][month_range][name]):
                                if (departure in duan) & (dest in duan):
                                    signal = 1
                                    break
                                if (departure in duan) ^ (dest in duan):
                                    return False
                        if signal == 0:
                            return False
        return True

    def bus_judge(bus_json):
        for m in bus_json['route']['transits'][0]['segments']:
            if 'bus' in m:
                return True
        return False

    def subway_mileage_cal(transit):
        mileage = 0
        for m in transit['segments']:
            if 'bus' in m:
                for busline in m['bus']['buslines']:
                    if busline['type'] == '地铁线路':
                        mileage += float(busline['distance'])
        return mileage

    if str((grid1, grid2)) in subway_json.keys():
        subway_exist = 0
        if len(subway_json[str((grid1, grid2))]['route']['transits']) > 0:
            for transit in subway_json[str((grid1, grid2))]['route']['transits']:
                if subway_exist == 0 and subway_judge(transit):
                    if not subway_history_judge(transit):
                        continue
                    subway_exist = 1
                    subway_duration = float(transit['duration'])
                    if len(transit['cost']) == 0:
                        subway_expense = 7
                    else:
                        subway_expense = float(transit['cost'])
                    subway_mileage = subway_mileage_cal(transit)
        if not subway_exist:
            subway_distance, subway_duration, subway_expense, subway_mileage = -1, -1, -1, -1

        # bus
        bus_exist = 0
        if len(subway_json[str((grid1, grid2))]['route']['transits']) > 0:
            for transit in subway_json[str((grid1, grid2))]['route']['transits']:
                if not subway_judge(transit) and bus_exist == 0:
                    for m in transit['segments']:
                        if 'bus' in m:
                            bus_duration = float(transit['duration'])
                            bus_exist = 1
                            if len(transit['cost']) == 0:
                                bus_expense = 5
                            else:
                                bus_expense = float(transit['cost'])
        else:
            bus_distance, bus_duration, bus_expense = -1, -1, -1
            bus_exist = 1

        if not bus_exist:
            if len(bus_json[str((grid1, grid2))]['route']['transits']) > 0 and bus_judge(bus_json[str((grid1, grid2))]):
                bus_duration = float(bus_json[str((grid1, grid2))]['route']['transits'][0]['duration'])
                if len(bus_json[str((grid1, grid2))]['route']['transits'][0]['cost']) == 0:
                    bus_expense = 5
                else:
                    bus_expense = float(bus_json[str((grid1, grid2))]['route']['transits'][0]['cost'])
            else:
                bus_distance, bus_duration, bus_expense = -1, -1, -1

        drive_distance = float(drive_json[str((grid1, grid2))]['route']['paths'][0]['distance'])
        drive_duration = float(drive_json[str((grid1, grid2))]['route']['paths'][0]['duration'])

        bicycle_duration = float(bicycle_json[str((grid1, grid2))]['data']['paths'][0]['duration'])
        bicycle_distance = float(bicycle_json[str((grid1, grid2))]['data']['paths'][0]['distance'])
        count = 0
        if key in commute2subway[year][month_range]:
            if subway_duration > 0:
                subway_duration = subway_duration * commute2subway[year][month_range][key]
                count = 1
        else:
            if subway_duration > 0:
                subway_duration = subway_duration * avecommute2subway[year][month_range]
        if key in commute2drive:
            drive_duration = drive_duration * commute2drive[key]
        else:
            drive_duration = drive_duration * avecommute2drive
        if key in commute2bicycle:
            bicycle_duration = bicycle_duration * commute2bicycle[key]
        else:
            bicycle_duration = bicycle_duration * avecommute2bicycle
        if key in commute2bus:
            bus_duration = bus_duration * commute2bus[key]
        else:
            bus_duration = bus_duration * avecommute2bus
        return subway_expense, bus_expense, drive_duration, subway_duration, bus_duration, bicycle_duration, bicycle_distance, drive_distance, subway_mileage, grid1, grid2, i, j, count
    if str((grid2, grid1)) in subway_json.keys():
        subway_exist = 0
        if len(subway_json[str((grid2, grid1))]['route']['transits']) > 0:
            for transit in subway_json[str((grid2, grid1))]['route']['transits']:
                if subway_exist == 0 and subway_judge(transit):
                    if not subway_history_judge(transit):
                        continue
                    subway_exist = 1
                    subway_duration = float(transit['duration'])
                    if len(transit['cost']) == 0:
                        subway_expense = 7
                    else:
                        subway_expense = float(transit['cost'])
                    subway_mileage = subway_mileage_cal(transit)
        if not subway_exist:
            subway_distance, subway_duration, subway_expense, subway_mileage = -1, -1, -1, -1

        # bus
        bus_exist = 0
        if len(subway_json[str((grid2, grid1))]['route']['transits']) > 0:
            for transit in subway_json[str((grid2, grid1))]['route']['transits']:
                if bus_exist == 0 and not subway_judge(transit):
                    for m in transit['segments']:
                        if 'bus' in m:
                            bus_duration = float(transit['duration'])
                            bus_exist = 1
                            if len(transit['cost']) == 0:
                                bus_expense = 5
                            else:
                                bus_expense = float(transit['cost'])
        else:
            bus_distance, bus_duration, bus_expense = -1, -1, -1
            bus_exist = 1

        if not bus_exist:
            if len(bus_json[str((grid2, grid1))]['route']['transits']) > 0 and bus_judge(bus_json[str((grid2, grid1))]):
                bus_duration = float(bus_json[str((grid2, grid1))]['route']['transits'][0]['duration'])
                if len(bus_json[str((grid2, grid1))]['route']['transits'][0]['cost']) == 0:
                    bus_expense = 5
                else:
                    bus_expense = float(bus_json[str((grid2, grid1))]['route']['transits'][0]['cost'])
            else:
                bus_distance, bus_duration, bus_expense = -1, -1, -1

        drive_distance = float(drive_json[str((grid2, grid1))]['route']['paths'][0]['distance'])
        drive_duration = float(drive_json[str((grid2, grid1))]['route']['paths'][0]['duration'])

        bicycle_duration = float(bicycle_json[str((grid2, grid1))]['data']['paths'][0]['duration'])
        bicycle_distance = float(bicycle_json[str((grid2, grid1))]['data']['paths'][0]['distance'])
        count = 0
        if key in commute2subway[year][month_range]:
            if subway_duration > 0:
                subway_duration = subway_duration * commute2subway[year][month_range][key]
                count = 1
        else:
            if subway_duration > 0:
                subway_duration = subway_duration * avecommute2subway[year][month_range]
        if key in commute2drive:
            drive_duration = drive_duration * commute2drive[key]
        else:
            drive_duration = drive_duration * avecommute2drive
        if key in commute2bicycle:
            bicycle_duration = bicycle_duration * commute2bicycle[key]
        else:
            bicycle_duration = bicycle_duration * avecommute2bicycle
        if key in commute2bus:
            bus_duration = bus_duration * commute2bus[key]
        else:
            bus_duration = bus_duration * avecommute2bus
        return subway_expense, bus_expense, drive_duration, subway_duration, bus_duration, bicycle_duration, bicycle_distance, drive_distance, subway_mileage, grid1, grid2, i, j, count

def yearinfo_collect(year):
    print(year)
    alpha = 0.08
    beta = 0.04
    od_matrix = load_npz('OD_{}/UO_od_WorldPop_{}.npz'.format(city_en_short, min(year, 2020))).todense().astype(
        np.float32).tolist()

    year2info = {}
    for months in year2station_operation[year]:
        year2info[months] = {}
        month_range = months
        month = eval(month_range)[0]
        strategy = 'First'
        subway_filtered_set = set([])
        available_subway = list(subway_years[subway_years['open_year'] < year]['地铁线路']) + list(
            subway_years[(subway_years['open_year'] == year) & (subway_years['open_month'] < month)]['地铁线路'])
        for subway in subway_set:
            if subway.split('线')[0] + '线' in available_subway:
                subway_filtered_set.add(subway)

        subway_expense_matrix = np.zeros((len(grids), len(grids)))
        bus_expense_matrix = np.zeros((len(grids), len(grids)))
        driving_duration_matrix = np.zeros((len(grids), len(grids)))
        subway_duration_matrix = np.zeros((len(grids), len(grids)))
        bus_duration_matrix = np.zeros((len(grids), len(grids)))
        bicycle_duration_matrix = np.zeros((len(grids), len(grids)))
        bicycle_distance_matrix = np.zeros((len(grids), len(grids)))
        driving_distance_matrix = np.zeros((len(grids), len(grids)))
        subway_mileage_matrix = np.zeros((len(grids), len(grids)))
        missed_od = 0

        args_list = []
        for i in tqdm(range(len(grids))):
            for j in range(i + 1, len(grids)):
                if od_matrix[i][j] >= 5 or od_matrix[j][i] >= 5:
                    args_list.append((list(grids[i]), list(grids[j]), i, j, subway_filtered_set, year, month_range))

        num_processes = 5
        pool = Pool(processes=num_processes)

        count = 0
        count_true = 0
        for result in tqdm(pool.imap(route, args_list), total=len(args_list)):
            if result is not None:
                subway_expense, bus_expense, driving_duration, subway_duration, bus_duration, bicycle_duration, bicycle_distance, driving_distance, subway_mileage, grid1, grid2, i, j, count_tmp = result
                count += 1
                count_true += count_tmp
                subway_expense_matrix[i, j] = subway_expense_matrix[j, i] = subway_expense
                bus_expense_matrix[i, j] = bus_expense_matrix[j, i] = bus_expense
                driving_duration_matrix[i, j] = driving_duration_matrix[j, i] = driving_duration
                subway_duration_matrix[i, j] = subway_duration_matrix[j, i] = subway_duration
                bicycle_duration_matrix[i, j] = bicycle_duration_matrix[j, i] = bicycle_duration
                bicycle_distance_matrix[i, j] = bicycle_distance_matrix[j, i] = bicycle_distance
                bus_duration_matrix[i, j] = bus_duration_matrix[j, i] = bus_duration
                driving_distance_matrix[i, j] = driving_distance_matrix[j, i] = driving_distance
                bicycle_distance_matrix[i, j] = bicycle_distance_matrix[j, i] = bicycle_distance
                subway_mileage_matrix[i, j] = subway_mileage_matrix[j, i] = subway_mileage
        pool.close()
        pool.join()
        logging.info('count, count_true: %s, %s', count, count_true)
        logging.info('len args: %s, missing rate: %s', len(args_list), (len(args_list) - count) / len(args_list))

        bus_expense_matrix = bus_expense_matrix / 4
        bicycle_duration_matrix = bicycle_duration_matrix / 1.5

        subway_distribution_dict = {}

        # commute part
        fuel_car_distance, elec_car_distance = 0, 0
        subway_save_distance = 0
        bus_num, subway_num, bicycle_num, drive_num = 0, 0, 0, 0
        subway_mileage_total = 0

        args_list = []
        for i in tqdm(range(len(grids))):
            for j in range(i + 1, len(grids)):
                grid0, grid1 = grids[i], grids[j]
                if (od_matrix[i][j] >= 5 or od_matrix[j][i] >= 5) and sum(
                        (abs(grid0[0] - grid1[0]), abs(grid0[1] - grid1[1]))) >= 1:
                    args_list.append((subway_expense_matrix[i][j], bus_expense_matrix[i][j],
                                      driving_duration_matrix[i][j], subway_duration_matrix[i][j],
                                      bus_duration_matrix[i][j], bicycle_duration_matrix[i][j],
                                      bicycle_distance_matrix[i][j],
                                      driving_distance_matrix[i][j], subway_mileage_matrix[i][j], od_matrix[i][j],
                                      od_matrix[j][i], grid0, grid1, year, month_range, subway_filtered_set))

        num_processes = 5
        pool = Pool(processes=num_processes)
        for result in pool.imap(get_mode_distribution, args_list):
            if result is not None:
                bus_ratio, subway_ratio, fuel_car_ratio, elec_car_ratio, bicycle_ratio, od1, od2, driving_distance, subway_mileage, grid0, grid1, subway_line_distribution = result
                od = 0
                if od1 >= 5:
                    od += od1
                if od2 >= 5:
                    od += od2
                fuel_car_distance += od * fuel_car_ratio * driving_distance * 2 * commute_ratio
                elec_car_distance += od * elec_car_ratio * driving_distance * 2 * commute_ratio
                bus_num += od * bus_ratio * 2 * commute_ratio
                subway_num += od * subway_ratio * 2 * commute_ratio
                drive_num += od * (fuel_car_ratio + elec_car_ratio) * 2 * commute_ratio
                bicycle_num += od * bicycle_ratio * 2 * commute_ratio
                subway_save_distance += od * subway_ratio * driving_distance * 2 * commute_ratio
                subway_mileage_total += od * subway_ratio * subway_mileage * 2 * commute_ratio
                for key in subway_line_distribution:
                    if key in subway_distribution_dict:
                        subway_distribution_dict[key] += od * subway_ratio * 2 * commute_ratio
                    else:
                        subway_distribution_dict[key] = od * subway_ratio * 2 * commute_ratio
        pool.close()
        pool.join()

        # carbon emission calculation
        fuel_emission_factor = 184  # gCO2/km
        elec_emission_factor = 78.7
        carbon = (fuel_car_distance * fuel_emission_factor + elec_car_distance * elec_emission_factor) / vehicle_occupancy / 1000

        year2info[months]['traffic_num'] = [subway_num, bus_num, drive_num, bicycle_num]
        year2info[months]['traffic_ratio'] = np.array(year2info[months]['traffic_num']) / sum(
            year2info[months]['traffic_num'])
        year2info[months]['traffic_ratio'] = year2info[months]['traffic_ratio'].tolist()
        year2info[months]['carbon'] = carbon
        year2info[months]['subway_distribution'] = subway_distribution_dict
        year2info[months]['fuel_car_distance'] = fuel_car_distance
        year2info[months]['elec_car_distance'] = elec_car_distance
        year2info[months]['subway_mileage'] = subway_mileage_total
    return year2info, year

year2info_years = {}

num_processes = args.num_process
pool = Pool(processes=num_processes)

args_list = list(range(2010, 2024))

for result in pool.imap(yearinfo_collect, args_list):
    if result is not None:
        year2info_years[result[-1]] = result[0]

pool.close()
pool.join()
logging.info('Factual data collection finished, %s' % str(year2info_years))

with open(basic_dir + '/years_results_overall_months_{}.json'.format(city_en_short), 'w') as f:
    json.dump(year2info_years, f)

with open(basic_dir + '/years_results_overall_months_{}.json'.format(city_en_short), 'r') as f:
    results = json.load(f)

## Fidelity check
if args.city_chn not in ['天津', '重庆', '南京', '长春', '大连', '苏州', '青岛']:
    official_stats = pd.read_excel('subway_boundary_file/official_statistics.xlsx', sheet_name=None)
    passenger_entry_volume_true_list = []
    passenger_traffic_volume_true_list = []
    passenger_transport_mileage_true_list = []

    for year in [2019, 2018, 2017, 2016, 2015]:
        tmp = official_stats[str(year)][official_stats[str(year)]['城市'] == city_chn]
        passenger_entry_volume_true_list.append(float(tmp['进站量 (万人次)'].sum()) / 365)
        passenger_traffic_volume_true_list.append(float(tmp['客运量 (万人次)'].sum()) / 365)
        passenger_transport_mileage_true_list.append(float(tmp['客运周转量 (万人次公里)'].sum()) / 365)

    passenger_entry_volume_list = []
    passenger_traffic_volume_list = []
    passenger_transport_mileage_list = []
    for year in [2019, 2018, 2017, 2016, 2015]:
        passenger_entry_volume, passenger_traffic_volume, passenger_transport_mileage = 0, 0, 0
        for months in results[str(year)]:
            passenger_entry_volume += results[str(year)][months]['traffic_num'][0] / (1e4) * len(eval(months))
            passenger_traffic_volume += np.sum(list(results[str(year)][months]['subway_distribution'].values())) / 1e4 * len(eval(months))
            passenger_transport_mileage += results[str(year)][months]['subway_mileage'] / (1e7) * len(eval(months))
        passenger_entry_volume_list.append(passenger_entry_volume / 12)
        passenger_traffic_volume_list.append(passenger_traffic_volume / 12)
        passenger_transport_mileage_list.append(passenger_transport_mileage / 12)
    ratio = passenger_entry_volume_true_list[0] / passenger_entry_volume_list[0]
    logging.info('passenger_entry_volume_true_list: %s' % str(passenger_entry_volume_true_list))
    logging.info('passenger_entry_volume_list: %s' % str(passenger_entry_volume_list))
    logging.info('passenger_traffic_volume_true_list: %s' % str(passenger_traffic_volume_true_list))
    logging.info('passenger_traffic_volume_list: %s' % str(passenger_traffic_volume_list))
    logging.info('passenger_transport_mileage_true_list: %s' % str(passenger_transport_mileage_true_list))
    logging.info('passenger_transport_mileage_list: %s' % str(passenger_transport_mileage_list))
    logging.info('ratio: %s' % str(ratio))
else:
    official_stats = pd.read_excel('subway_boundary_file/official_statistics_2019.xlsx', sheet_name=None)
    passenger_traffic_volume_true_list = []
    passenger_transport_mileage_true_list = []

    for year in [2019, 2018]:
        tmp = official_stats[str(year)][official_stats[str(year)]['城市'] == city_chn]
        passenger_traffic_volume_true_list.append(float(tmp['客运量 (万人次)'].sum()) / 365)
        passenger_transport_mileage_true_list.append(float(tmp['客运周转量 (万人次公里)'].sum()) / 365)

    passenger_traffic_volume_list = []
    passenger_transport_mileage_list = []
    for year in [2019, 2018, 2017, 2016, 2015]:
        passenger_entry_volume, passenger_traffic_volume, passenger_transport_mileage = 0, 0, 0
        for months in results[str(year)]:
            passenger_entry_volume += results[str(year)][months]['traffic_num'][0] / (1e4) * len(eval(months))
            passenger_traffic_volume += np.sum(list(results[str(year)][months]['subway_distribution'].values())) / 1e4 * len(eval(months))
            passenger_transport_mileage += results[str(year)][months]['subway_mileage'] / (1e7) * len(eval(months))
        passenger_traffic_volume_list.append(passenger_traffic_volume / 12)
        passenger_transport_mileage_list.append(passenger_transport_mileage / 12)
    ratio = passenger_traffic_volume_true_list[0] / passenger_traffic_volume_list[0]
    logging.info('passenger_traffic_volume_true_list: %s' % str(passenger_traffic_volume_true_list))
    logging.info('passenger_traffic_volume_list: %s' % str(passenger_traffic_volume_list))
    logging.info('passenger_transport_mileage_true_list: %s' % str(passenger_transport_mileage_true_list))
    logging.info('passenger_transport_mileage_list: %s' % str(passenger_transport_mileage_list))
    logging.info('ratio: %s' % str(ratio))  

if args.city_chn not in ['天津', '重庆', '南京', '长春', '大连', '苏州', '青岛']:
    year_list = ['2019', '2018', '2017', '2016', '2015']
    mpl.rcParams['xtick.major.pad'] = 10
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.sans-serif'] = 'Times New Roman'

    ### passenger entry volume
    plt.figure(figsize=(12, 9))
    plt.rcParams['savefig.dpi'] = 100  
    plt.rcParams['figure.dpi'] = 100  

    y1 = np.array(passenger_entry_volume_list)[:len(year_list)] * ratio
    y2 = np.array(passenger_entry_volume_true_list)
    categories = year_list

    bar_width = 0.3
    x1 = np.arange(len(y1))
    x2 = [x + bar_width for x in x1]

    plt.bar(x1, y1, width=bar_width, label='Simulated', color='indianred', alpha=0.9)
    plt.bar(x2, y2, width=bar_width, label='True', alpha=0.9)

    plt.xticks([x + bar_width for x in x1], categories, fontsize=16)

    for i, v in enumerate(y1):
        plt.text(x1[i], v, str(int(v)), ha='center', va='bottom')

    for i, v in enumerate(y2):
        plt.text(x2[i], v, str(int(v)), ha='center', va='bottom')

    plt.yticks(fontsize=16)
    plt.ylabel('Total Passengers (1e4)', fontsize=16)

    plt.legend(fontsize=16)

    plt.savefig(basic_dir + '/passenger_entry_volume.jpg', format="png", bbox_inches="tight")

    year_list = ['2019', '2018', '2017', '2016', '2015']
    ### passenger traffic volume
    plt.figure(figsize=(12, 9))
    plt.rcParams['savefig.dpi'] = 100 
    plt.rcParams['figure.dpi'] = 100  

    y1 = np.array(passenger_traffic_volume_list)[:len(year_list)] * ratio
    y2 = np.array(passenger_traffic_volume_true_list)
    categories = year_list

    bar_width = 0.3
    x1 = np.arange(len(y1))
    x2 = [x + bar_width for x in x1]

    plt.bar(x1, y1, width=bar_width, label='Simulated', color='indianred', alpha=0.9)
    plt.bar(x2, y2, width=bar_width, label='True', alpha=0.9)

    plt.xticks([x + bar_width for x in x1], categories, fontsize=16)

    for i, v in enumerate(y1):
        plt.text(x1[i], v, str(int(v)), ha='center', va='bottom')

    for i, v in enumerate(y2):
        plt.text(x2[i], v, str(int(v)), ha='center', va='bottom')

    plt.yticks(fontsize=16)
    plt.ylabel('Total Flow（1e4）', fontsize=16)

    plt.legend(fontsize=16)

    plt.savefig(basic_dir + '/passenger_traffic_volume.jpg', format="png", bbox_inches="tight")

    ### passenger transport mileage
    year_list = ['2019', '2018', '2017', '2016', '2015']
    plt.figure(figsize=(12, 9))
    plt.rcParams['savefig.dpi'] = 100  
    plt.rcParams['figure.dpi'] = 100  

    y1 = np.array(passenger_transport_mileage_list)[:len(year_list)] * ratio
    y2 = np.array(passenger_transport_mileage_true_list)
    categories = year_list

    bar_width = 0.3
    x1 = np.arange(len(y1))
    x2 = [x + bar_width for x in x1]

    plt.bar(x1, y1, width=bar_width, label='Simulated', color='indianred', alpha=0.9)
    plt.bar(x2, y2, width=bar_width, label='True', alpha=0.9)

    plt.xticks([x + bar_width for x in x1], categories, fontsize=16)

    for i, v in enumerate(y1):
        plt.text(x1[i], v, str(int(v)), ha='center', va='bottom')

    for i, v in enumerate(y2):
        plt.text(x2[i], v, str(int(v)), ha='center', va='bottom')

    plt.yticks(fontsize=16)
    plt.ylabel('Total Mileage (1e4km)', fontsize=16)

    plt.legend(fontsize=16)
    plt.savefig(basic_dir + '/passenger_transport_mileage.jpg', format="png", bbox_inches="tight")
else:
    year_list = ['2019', '2018']
    ### passenger traffic volume
    plt.figure(figsize=(12, 9))
    plt.rcParams['savefig.dpi'] = 100 
    plt.rcParams['figure.dpi'] = 100  

    y1 = np.array(passenger_traffic_volume_list)[:len(year_list)] * ratio
    y2 = np.array(passenger_traffic_volume_true_list)
    categories = year_list

    bar_width = 0.3
    x1 = np.arange(len(y1))
    x2 = [x + bar_width for x in x1]

    plt.bar(x1, y1, width=bar_width, label='Simulated', color='indianred', alpha=0.9)
    plt.bar(x2, y2, width=bar_width, label='True', alpha=0.9)

    plt.xticks([x + bar_width for x in x1], categories, fontsize=16)

    for i, v in enumerate(y1):
        plt.text(x1[i], v, str(int(v)), ha='center', va='bottom')

    for i, v in enumerate(y2):
        plt.text(x2[i], v, str(int(v)), ha='center', va='bottom')

    plt.yticks(fontsize=16)
    plt.ylabel('Total Flow（1e4）', fontsize=16)

    plt.legend(fontsize=16)

    plt.savefig(basic_dir + '/passenger_traffic_volume.jpg', format="png", bbox_inches="tight")

    ### passenger transport mileage
    year_list = ['2019', '2018']
    plt.figure(figsize=(12, 9))
    plt.rcParams['savefig.dpi'] = 100  
    plt.rcParams['figure.dpi'] = 100  

    y1 = np.array(passenger_transport_mileage_list)[:len(year_list)] * ratio
    y2 = np.array(passenger_transport_mileage_true_list)
    categories = year_list

    bar_width = 0.3
    x1 = np.arange(len(y1))
    x2 = [x + bar_width for x in x1]

    plt.bar(x1, y1, width=bar_width, label='Simulated', color='indianred', alpha=0.9)
    plt.bar(x2, y2, width=bar_width, label='True', alpha=0.9)

    plt.xticks([x + bar_width for x in x1], categories, fontsize=16)

    for i, v in enumerate(y1):
        plt.text(x1[i], v, str(int(v)), ha='center', va='bottom')

    for i, v in enumerate(y2):
        plt.text(x2[i], v, str(int(v)), ha='center', va='bottom')

    plt.yticks(fontsize=16)
    plt.ylabel('Total Mileage (1e4km)', fontsize=16)

    plt.legend(fontsize=16)
    plt.savefig(basic_dir + '/passenger_transport_mileage.jpg', format="png", bbox_inches="tight")

print('5. Counterfactual simulation for subway line removal')
construction_df = pd.read_csv('result_files/subway_construction_stats_2019.csv') 
construction_df = construction_df[construction_df['City']==city_en]

subway2open_year = {}
for i in range(len(subway_years)):
    line = subway_years.iloc[i]['地铁线路']
    year = subway_years.iloc[i]['open_year']
    if line in subway2open_year:
        if year < subway2open_year[line]:
            subway2open_year[line] = year
    else:
        subway2open_year[line] = year

remove_subway_list = [line for line in subway2open_year if subway2open_year[line] < 2020]
logging.info('remove_subway_list: {}'.format(remove_subway_list))

def yearinfo_collect_remove(args):
    year, remove_subway = args[0], args[1]
    print(year)
    od_matrix = load_npz('OD_{}/UO_od_WorldPop_{}.npz'.format(city_en_short, min(year, 2020))).todense().astype(np.float32).tolist()

    year2info = {}
    for months in year2station_operation[year]:
        year2info[months] = {}
        month_range = months
        month = eval(month_range)[0]
        strategy = 'First'
        subway_filtered_set = set([])
        available_subway = list(subway_years[subway_years['open_year'] < year]['地铁线路'])+ list(subway_years[(subway_years['open_year'] == year) & (subway_years['open_month'] < month)]['地铁线路'])
        for subway in subway_set:
            if subway.split('线')[0]+'线' == remove_subway:
                continue
            if subway.split('线')[0] + '线' in available_subway:
                subway_filtered_set.add(subway)

        subway_expense_matrix = np.zeros((len(grids), len(grids)))
        bus_expense_matrix = np.zeros((len(grids), len(grids)))
        driving_duration_matrix = np.zeros((len(grids), len(grids)))
        subway_duration_matrix = np.zeros((len(grids), len(grids)))
        bus_duration_matrix = np.zeros((len(grids), len(grids)))
        bicycle_duration_matrix = np.zeros((len(grids), len(grids)))
        bicycle_distance_matrix = np.zeros((len(grids), len(grids)))
        driving_duration_matrix_noncommute = np.zeros((len(grids), len(grids)))
        subway_duration_matrix_noncommute = np.zeros((len(grids), len(grids)))
        bus_duration_matrix_noncommute = np.zeros((len(grids), len(grids)))
        bicycle_duration_matrix_noncommute = np.zeros((len(grids), len(grids)))
        driving_distance_matrix = np.zeros((len(grids), len(grids)))
        subway_mileage_matrix = np.zeros((len(grids), len(grids)))
        missed_od = 0

        args_list = []
        for i in tqdm(range(len(grids))):
            for j in range(i + 1, len(grids)):
                if od_matrix[i][j] >= 5 or od_matrix[j][i] >= 5:
                    args_list.append((list(grids[i]), list(grids[j]), i, j, subway_filtered_set, year, month_range))

        num_processes = 3
        pool = Pool(processes=num_processes)

        count = 0
        count_true = 0
        for result in tqdm(pool.imap(route, args_list), total=len(args_list)):
            if result is not None:
                subway_expense, bus_expense, driving_duration, subway_duration, bus_duration, bicycle_duration, bicycle_distance, driving_distance, subway_mileage, grid1, grid2, i, j, count_tmp = result
                count += 1
                count_true += count_tmp
                bicycle_distance_matrix[i, j] = bicycle_distance_matrix[j, i] = bicycle_distance
                subway_expense_matrix[i, j] = subway_expense_matrix[j, i] = subway_expense
                bus_expense_matrix[i, j] = bus_expense_matrix[j, i] = bus_expense
                driving_duration_matrix[i, j] = driving_duration_matrix[j, i] = driving_duration
                subway_duration_matrix[i, j] = subway_duration_matrix[j, i] = subway_duration
                bicycle_duration_matrix[i, j] = bicycle_duration_matrix[j, i] = bicycle_duration
                bus_duration_matrix[i, j] = bus_duration_matrix[j, i] = bus_duration
                driving_distance_matrix[i, j] = driving_distance_matrix[j, i] = driving_distance
                subway_mileage_matrix[i, j] = subway_mileage_matrix[j, i] = subway_mileage
        pool.close()
        pool.join()
        logging.info('count, count_true: %s, %s', count, count_true)

        bus_expense_matrix = bus_expense_matrix / 4
        bicycle_duration_matrix = bicycle_duration_matrix / 1.5

        subway_distribution_dict = {}

        # commuting
        fuel_car_distance, elec_car_distance = 0, 0
        subway_save_distance = 0
        bus_num, subway_num, bicycle_num, drive_num = 0, 0, 0, 0
        subway_mileage_total = 0

        args_list = []
        for i in tqdm(range(len(grids))):
            for j in range(i + 1, len(grids)):
                grid0, grid1 = grids[i], grids[j]
                if (od_matrix[i][j] >= 5 or od_matrix[j][i] >= 5) and sum(
                        (abs(grid0[0] - grid1[0]), abs(grid0[1] - grid1[1]))) >= 1:
                    args_list.append((subway_expense_matrix[i][j], bus_expense_matrix[i][j],
                              driving_duration_matrix[i][j], subway_duration_matrix[i][j],
                              bus_duration_matrix[i][j], bicycle_duration_matrix[i][j], bicycle_distance_matrix[i][j],
                              driving_distance_matrix[i][j], subway_mileage_matrix[i][j], od_matrix[i][j],
                              od_matrix[j][i], grid0, grid1, year, month_range, subway_filtered_set))

        num_processes = 5
        pool = Pool(processes=num_processes)
        for result in pool.imap(get_mode_distribution, args_list):
            if result is not None:
                bus_ratio, subway_ratio, fuel_car_ratio, elec_car_ratio, bicycle_ratio, od1, od2, driving_distance, subway_mileage, grid0, grid1, subway_line_distribution = result
                od = 0
                if od1 >= 5:
                    od += od1
                if od2 >= 5:
                    od += od2
                fuel_car_distance += od * fuel_car_ratio * driving_distance * 2 * commute_ratio
                elec_car_distance += od * elec_car_ratio * driving_distance * 2 * commute_ratio
                bus_num += od * bus_ratio * 2 * commute_ratio
                subway_num += od * subway_ratio * 2 * commute_ratio
                drive_num += od * (fuel_car_ratio + elec_car_ratio) * 2 * commute_ratio
                bicycle_num += od * bicycle_ratio * 2 * commute_ratio
                subway_save_distance += od * subway_ratio * driving_distance * 2 * commute_ratio
                subway_mileage_total += od * subway_ratio * subway_mileage * 2 * commute_ratio
                for key in subway_line_distribution:
                    if key in subway_distribution_dict:
                        subway_distribution_dict[key] += od * subway_ratio * 2 * commute_ratio
                    else:
                        subway_distribution_dict[key] = od * subway_ratio * 2 * commute_ratio
        pool.close()
        pool.join()

        # carbon emission calculation
        fuel_emission_factor = 184  # gCO2/km
        elec_emission_factor = 78.7
        carbon = (fuel_car_distance * fuel_emission_factor + elec_car_distance * elec_emission_factor) / vehicle_occupancy/ 1000

        year2info[months]['traffic_num'] = [subway_num, bus_num, drive_num, bicycle_num]
        year2info[months]['traffic_ratio'] = np.array(year2info[months]['traffic_num']) / sum(year2info[months]['traffic_num'])
        year2info[months]['traffic_ratio'] = year2info[months]['traffic_ratio'].tolist()
        year2info[months]['carbon'] = carbon
        year2info[months]['subway_distribution'] = subway_distribution_dict
        year2info[months]['fuel_car_distance'] = fuel_car_distance
        year2info[months]['elec_car_distance'] = elec_car_distance
        year2info[months]['subway_mileage'] = subway_mileage_total
    return year2info, year

for remove_subway in remove_subway_list:
    print('Remove: ', remove_subway)
    try:
        if '铁' in remove_subway:
            remove_subway_num = remove_subway.split('号线')[0].split('铁')[1]
        elif '轨' in remove_subway:
            remove_subway_num = remove_subway.split('号线')[0].split('通')[1]
        else:
            pinyin_result = pinyin(remove_subway, style=Style.NORMAL)
            remove_subway_num = ' '.join([''.join(item) for item in pinyin_result])
    except:
        continue
    year2info_years = {}

    num_processes = args.num_process
    pool = Pool(processes=num_processes)

    args_list = [(year, remove_subway) for year in range(2010, 2024)]

    for result in pool.imap(yearinfo_collect_remove, args_list):
        if result is not None:
                year2info_years[result[-1]] = result[0]
    pool.close()
    pool.join()

    with open(basic_dir+'/years_results_fenduan_without_{}.json'.format(remove_subway), 'w') as f:
        json.dump(year2info_years, f)

print('6. Counterfactual simulation for subway system removal')
min_open_year = min(subway_years['open_year'])
if min_open_year < 2010:
    last_year = 2009
else:
    last_year = min_open_year - 1

remove_subway_list = set(subway_years[subway_years['remove'] == 1]['地铁线路'])
remaining_subway_list = set(subway_years['地铁线路']) - remove_subway_list

subway_filtered_set = set([])
available_subway = list(subway_years[subway_years['open_year'] < 2010]['地铁线路'])
for subway in subway_set:
    if subway.split('线')[0] + '线' in available_subway:
        subway_filtered_set.add(subway)
for subway in remove_subway_list:
    subway_filtered_set.add(subway)

if last_year == 2009:
    month_range = [month for month in subway_info['2010'] if '1' in month][0]
    existing_subways = subway_info[str(2010)][month_range]
else:
    existing_subways = {}

counterfactual_subway_info = {}
for year in range(2010, 2024):
    counterfactual_subway_info[str(year)] = {}
    if str(year) in subway_info:
        tmp = [month for month in subway_info[str(year)] if '1' in month]
        if len(tmp) == 0:
            continue
        month_range = tmp[0]
        for line in subway_info[str(year)][month_range]:
            if line in remove_subway_list:
                counterfactual_subway_info[str(year)][line] = subway_info[str(year)][month_range][line]
            else:
                if line in existing_subways:
                    counterfactual_subway_info[str(year)][line] = existing_subways[line]

construction_df = pd.read_csv('result_files/subway_construction_stats_2019.csv')
construction_df = construction_df[construction_df['City'] == city_en]

def route(args):
    grid1, grid2, i, j, subway_filtered_set, year, month_range = args
    grid1, grid2 = list(grid1), list(grid2)

    large_grid1 = grid2large_grid[tuple(grid1)]
    large_grid2 = grid2large_grid[tuple(grid2)]
    key = (0, 0)
    if (large_grid1, large_grid2) in large_grid_pairs:
        key = str((large_grid1, large_grid2))
    if (large_grid2, large_grid1) in large_grid_pairs:
        key = str((large_grid2, large_grid1))

    month_range = [month for month in subway_info['2020'] if '1' in month][0]

    def subway_judge(transit):
        for m in transit['segments']:
            if 'bus' in m:
                for busline in m['bus']['buslines']:
                    if busline['type'] == '地铁线路':
                        return True
        return False

    def subway_history_judge(transit):
        for m in transit['segments']:
            if 'bus' in m:
                for busline in m['bus']['buslines']:
                    if busline['type'] == '地铁线路':
                        if busline['name'] not in subway_filtered_set:
                            return False
                        signal = 0
                        name = busline['name'].split('线')[0] + '线'
                        if name not in counterfactual_subway_info[str(year)][name]:
                            signal = 1
                        else:
                            departure = busline['departure_stop']['name']
                            dest = busline['arrival_stop']['name']
                            for duan in eval(counterfactual_subway_info[str(year)][name]):
                                if (departure in duan) & (dest in duan):
                                    signal = 1
                                    break
                                if (departure in duan) ^ (dest in duan):
                                    return False
                        if signal == 0:
                            return False
        return True

    def bus_judge(bus_json):
        for m in bus_json['route']['transits'][0]['segments']:
            if 'bus' in m:
                return True
        return False

    def subway_mileage_cal(transit):
        mileage = 0
        for m in transit['segments']:
            if 'bus' in m:
                for busline in m['bus']['buslines']:
                    if busline['type'] == '地铁线路':
                        mileage += float(busline['distance'])
        return mileage

    if str((grid1, grid2)) in subway_json.keys():
        subway_exist = 0
        if len(subway_json[str((grid1, grid2))]['route']['transits']) > 0:
            for transit in subway_json[str((grid1, grid2))]['route']['transits']:
                if subway_exist == 0 and subway_judge(transit):
                    if not subway_history_judge(transit):
                        continue
                    subway_exist = 1
                    subway_duration = float(transit['duration'])
                    if len(transit['cost']) == 0:
                        subway_expense = 7
                    else:
                        subway_expense = float(transit['cost'])
                    subway_mileage = subway_mileage_cal(transit)
        if not subway_exist:
            subway_distance, subway_duration, subway_expense, subway_mileage = -1, -1, -1, -1

        # bus
        bus_exist = 0
        if len(subway_json[str((grid1, grid2))]['route']['transits']) > 0:
            for transit in subway_json[str((grid1, grid2))]['route']['transits']:
                if not subway_judge(transit) and bus_exist == 0:
                    for m in transit['segments']:
                        if 'bus' in m:
                            bus_duration = float(transit['duration'])
                            bus_exist = 1
                            if len(transit['cost']) == 0:
                                bus_expense = 5
                            else:
                                bus_expense = float(transit['cost'])
        else:
            bus_distance, bus_duration, bus_expense = -1, -1, -1
            bus_exist = 1

        if not bus_exist:
            if len(bus_json[str((grid1, grid2))]['route']['transits']) > 0 and bus_judge(bus_json[str((grid1, grid2))]):
                bus_duration = float(bus_json[str((grid1, grid2))]['route']['transits'][0]['duration'])
                if len(bus_json[str((grid1, grid2))]['route']['transits'][0]['cost']) == 0:
                    bus_expense = 5
                else:
                    bus_expense = float(bus_json[str((grid1, grid2))]['route']['transits'][0]['cost'])
            else:
                bus_distance, bus_duration, bus_expense = -1, -1, -1

        drive_distance = float(drive_json[str((grid1, grid2))]['route']['paths'][0]['distance'])
        drive_duration = float(drive_json[str((grid1, grid2))]['route']['paths'][0]['duration'])

        bicycle_duration = float(bicycle_json[str((grid1, grid2))]['data']['paths'][0]['duration'])
        bicycle_distance = float(bicycle_json[str((grid1, grid2))]['data']['paths'][0]['distance'])
        count = 0
        if key in commute2subway[2020][month_range]:
            if subway_duration > 0:
                subway_duration = subway_duration * commute2subway[2020][month_range][key]
                count = 1
        else:
            if subway_duration > 0:
                subway_duration = subway_duration * avecommute2subway[2020][month_range]
        if key in commute2drive:
            drive_duration = drive_duration * commute2drive[key]
        else:
            drive_duration = drive_duration * avecommute2drive
        if key in commute2bicycle:
            bicycle_duration = bicycle_duration * commute2bicycle[key]
        else:
            bicycle_duration = bicycle_duration * avecommute2bicycle
        if key in commute2bus:
            bus_duration = bus_duration * commute2bus[key]
        else:
            bus_duration = bus_duration * avecommute2bus
        return subway_expense, bus_expense, drive_duration, subway_duration, bus_duration, bicycle_duration, bicycle_distance, drive_distance, subway_mileage, grid1, grid2, i, j, count
    if str((grid2, grid1)) in subway_json.keys():
        subway_exist = 0
        if len(subway_json[str((grid2, grid1))]['route']['transits']) > 0:
            for transit in subway_json[str((grid2, grid1))]['route']['transits']:
                if subway_exist == 0 and subway_judge(transit):
                    if not subway_history_judge(transit):
                        continue
                    subway_exist = 1
                    subway_duration = float(transit['duration'])
                    if len(transit['cost']) == 0:
                        subway_expense = 7
                    else:
                        subway_expense = float(transit['cost'])
                    subway_mileage = subway_mileage_cal(transit)
        if not subway_exist:
            subway_distance, subway_duration, subway_expense, subway_mileage = -1, -1, -1, -1

        # bus
        bus_exist = 0
        if len(subway_json[str((grid2, grid1))]['route']['transits']) > 0:
            for transit in subway_json[str((grid2, grid1))]['route']['transits']:
                if bus_exist == 0 and not subway_judge(transit):
                    for m in transit['segments']:
                        if 'bus' in m:
                            bus_duration = float(transit['duration'])
                            bus_exist = 1
                            if len(transit['cost']) == 0:
                                bus_expense = 5
                            else:
                                bus_expense = float(transit['cost'])
        else:
            bus_distance, bus_duration, bus_expense = -1, -1, -1
            bus_exist = 1

        if not bus_exist:
            if len(bus_json[str((grid2, grid1))]['route']['transits']) > 0 and bus_judge(bus_json[str((grid2, grid1))]):
                bus_duration = float(bus_json[str((grid2, grid1))]['route']['transits'][0]['duration'])
                if len(bus_json[str((grid2, grid1))]['route']['transits'][0]['cost']) == 0:
                    bus_expense = 5
                else:
                    bus_expense = float(bus_json[str((grid2, grid1))]['route']['transits'][0]['cost'])
            else:
                bus_distance, bus_duration, bus_expense = -1, -1, -1

        drive_distance = float(drive_json[str((grid2, grid1))]['route']['paths'][0]['distance'])
        drive_duration = float(drive_json[str((grid2, grid1))]['route']['paths'][0]['duration'])

        bicycle_duration = float(bicycle_json[str((grid2, grid1))]['data']['paths'][0]['duration'])
        bicycle_distance = float(bicycle_json[str((grid2, grid1))]['data']['paths'][0]['distance'])
        count = 0
        if key in commute2subway[2020][month_range]:
            if subway_duration > 0:
                subway_duration = subway_duration * commute2subway[2020][month_range][key]
                count = 1
        else:
            if subway_duration > 0:
                subway_duration = subway_duration * avecommute2subway[2020][month_range]
        if key in commute2drive:
            drive_duration = drive_duration * commute2drive[key]
        else:
            drive_duration = drive_duration * avecommute2drive
        if key in commute2bicycle:
            bicycle_duration = bicycle_duration * commute2bicycle[key]
        else:
            bicycle_duration = bicycle_duration * avecommute2bicycle
        if key in commute2bus:
            bus_duration = bus_duration * commute2bus[key]
        else:
            bus_duration = bus_duration * avecommute2bus
        return subway_expense, bus_expense, drive_duration, subway_duration, bus_duration, bicycle_duration, bicycle_distance, drive_distance, subway_mileage, grid1, grid2, i, j, count

def yearinfo_collect_city_counterfactual(year):
    print(year)
    od_matrix = load_npz(
        'OD_{}/UO_od_WorldPop_{}_counterfactual.npz'.format(city_en_short, min(year, 2020))).todense().astype(
        np.float32).tolist()

    year2info = {}

    subway_expense_matrix = np.zeros((len(grids), len(grids)))
    bus_expense_matrix = np.zeros((len(grids), len(grids)))
    driving_duration_matrix = np.zeros((len(grids), len(grids)))
    subway_duration_matrix = np.zeros((len(grids), len(grids)))
    bus_duration_matrix = np.zeros((len(grids), len(grids)))
    bicycle_duration_matrix = np.zeros((len(grids), len(grids)))
    bicycle_distance_matrix = np.zeros((len(grids), len(grids)))
    driving_duration_matrix_noncommute = np.zeros((len(grids), len(grids)))
    subway_duration_matrix_noncommute = np.zeros((len(grids), len(grids)))
    bus_duration_matrix_noncommute = np.zeros((len(grids), len(grids)))
    bicycle_duration_matrix_noncommute = np.zeros((len(grids), len(grids)))
    driving_distance_matrix = np.zeros((len(grids), len(grids)))
    subway_mileage_matrix = np.zeros((len(grids), len(grids)))
    missed_od = 0

    args_list = []
    for i in tqdm(range(len(grids))):
        for j in range(i + 1, len(grids)):
            if od_matrix[i][j] >= 5 or od_matrix[j][i] >= 5:
                args_list.append((list(grids[i]), list(grids[j]), i, j, subway_filtered_set, year, month_range))

    num_processes = 10
    pool = Pool(processes=num_processes)

    count = 0
    count_true = 0
    for result in tqdm(pool.imap(route, args_list), total=len(args_list)):
        if result is not None:
            subway_expense, bus_expense, driving_duration, subway_duration, bus_duration, bicycle_duration, bicycle_distance, driving_distance, subway_mileage, grid1, grid2, i, j, count_tmp = result
            count += 1
            count_true += count_tmp
            subway_expense_matrix[i, j] = subway_expense_matrix[j, i] = subway_expense
            bus_expense_matrix[i, j] = bus_expense_matrix[j, i] = bus_expense
            driving_duration_matrix[i, j] = driving_duration_matrix[j, i] = driving_duration
            subway_duration_matrix[i, j] = subway_duration_matrix[j, i] = subway_duration
            bicycle_duration_matrix[i, j] = bicycle_duration_matrix[j, i] = bicycle_duration
            bicycle_distance_matrix[i, j] = bicycle_distance_matrix[j, i] = bicycle_distance
            bus_duration_matrix[i, j] = bus_duration_matrix[j, i] = bus_duration
            driving_distance_matrix[i, j] = driving_distance_matrix[j, i] = driving_distance
            subway_mileage_matrix[i, j] = subway_mileage_matrix[j, i] = subway_mileage
    pool.close()
    pool.join()
    logging.info('count, count_true: %s, %s', count, count_true)

    bus_expense_matrix = bus_expense_matrix / 4
    bicycle_duration_matrix = bicycle_duration_matrix / 1.5

    def subway_line_distribute(grid1, grid2):
        grid1, grid2 = list(grid1), list(grid2)

        def subway_judge(transit):
            for m in transit['segments']:
                if 'bus' in m:
                    for busline in m['bus']['buslines']:
                        if busline['type'] == '地铁线路':
                            return True
            return False

        def subway_history_judge(transit):
            for m in transit['segments']:
                if 'bus' in m:
                    for busline in m['bus']['buslines']:
                        if busline['type'] == '地铁线路':
                            if busline['name'] not in subway_filtered_set:
                                return False
                            signal = 0
                            name = busline['name'].split('线')[0] + '线'
                            if name not in counterfactual_subway_info[str(year)][name]:
                                signal = 1
                            else:
                                departure = busline['departure_stop']['name']
                                dest = busline['arrival_stop']['name']
                                for duan in eval(counterfactual_subway_info[str(year)][name]):
                                    if (departure in duan) & (dest in duan):
                                        signal = 1
                                        break
                                    if (departure in duan) ^ (dest in duan):
                                        return False
                            if signal == 0:
                                return False
            return True

        def subway_dict_return(transit):
            subway_dict = {}
            for m in transit['segments']:
                if 'bus' in m:
                    for busline in m['bus']['buslines']:
                        if busline['type'] == '地铁线路':
                            subway_dict[busline['name']] = busline['distance'] if busline[
                                                                                      'name'] not in subway_dict else \
                                busline['distance'] + subway_dict[busline['name']]
            return subway_dict

        if str((grid1, grid2)) in subway_json.keys():
            if len(subway_json[str((grid1, grid2))]['route']['transits']) > 0:
                for transit in subway_json[str((grid1, grid2))]['route']['transits']:
                    if subway_judge(transit):
                        if not subway_history_judge(transit):
                            continue
                        subway_dict = subway_dict_return(transit)
                        return subway_dict
        if str((grid2, grid1)) in subway_json.keys():
            if len(subway_json[str((grid2, grid1))]['route']['transits']) > 0:
                for transit in subway_json[str((grid2, grid1))]['route']['transits']:
                    if subway_judge(transit):
                        if not subway_history_judge(transit):
                            continue
                        subway_dict = subway_dict_return(transit)
                        return subway_dict
        return {}

    def get_mode_distribution(args):
        subway_expense, bus_expense, driving_duration, subway_duration, bus_duration, bicycle_duration, distance, driving_distance, subway_mileage, od1, od2, grid0, grid1, year, month_range, subway_filtered_set = args
        if bicycle_duration == 0:
            return
        parking_fee = parking_fee_overall  # parking fee
        age = 0.384  # proportion of residents in age group 18-35
        income = 0.395  # low-income group ratio
        if bus_expense > 0:
            V_bus = -0.0516 * bus_duration / 60 - 0.4810 * bus_expense
        else:
            V_bus = -np.inf
        if subway_expense > 0:
            V_subway = -0.0512 * subway_duration / 60 - 0.0833 * subway_expense
        else:
            V_subway = -np.inf
        V_fuel = -0.0705 * driving_duration / 60 + 0.5680 * age - 0.8233 * income - 0.0941 * parking_fee
        V_elec = -0.0339 * driving_duration / 60 - 0.1735 * parking_fee
        if distance > 15000:  # no bicycling for long distance
            V_bicycle = -np.inf
        else:
            V_bicycle = -0.1185 * bicycle_duration / 60
        V = np.array([V_bus, V_subway, V_fuel, V_elec, V_bicycle])
        V = np.exp(V)
        V = V / sum(V)

        subway_line_distribution = subway_line_distribute(grid0, grid1)
        return V[0], V[1], V[2], V[3], V[
            4], od1, od2, driving_distance, subway_mileage, grid0, grid1, subway_line_distribution

    subway_distribution_dict = {}

    fuel_car_distance, elec_car_distance = 0, 0
    subway_save_distance = 0
    bus_num, subway_num, bicycle_num, drive_num = 0, 0, 0, 0
    subway_mileage_total = 0

    args_list = []
    for i in tqdm(range(len(grids))):
        for j in range(i + 1, len(grids)):
            grid0, grid1 = grids[i], grids[j]
            if (od_matrix[i][j] >= 5 or od_matrix[j][i] >= 5) and sum(
                    (abs(grid0[0] - grid1[0]), abs(grid0[1] - grid1[1]))) >= 1:
                args_list.append((subway_expense_matrix[i][j], bus_expense_matrix[i][j],
                                  driving_duration_matrix[i][j], subway_duration_matrix[i][j],
                                  bus_duration_matrix[i][j], bicycle_duration_matrix[i][j],
                                  bicycle_distance_matrix[i][j],
                                  driving_distance_matrix[i][j], subway_mileage_matrix[i][j], od_matrix[i][j],
                                  od_matrix[j][i], grid0, grid1, year, month_range, subway_filtered_set))

    for args in args_list:
        result = get_mode_distribution(args)
        if result is None:
            continue
        bus_ratio, subway_ratio, fuel_car_ratio, elec_car_ratio, bicycle_ratio, od1, od2, driving_distance, subway_mileage, grid0, grid1, subway_line_distribution = result
        od = 0
        if od1 >= 5:
            od += od1
        if od2 >= 5:
            od += od2
        fuel_car_distance += od * fuel_car_ratio * driving_distance * 2 * commute_ratio
        elec_car_distance += od * elec_car_ratio * driving_distance * 2 * commute_ratio
        bus_num += od * bus_ratio * 2 * commute_ratio
        subway_num += od * subway_ratio * 2 * commute_ratio
        drive_num += od * (fuel_car_ratio + elec_car_ratio) * 2 * commute_ratio
        bicycle_num += od * bicycle_ratio * 2 * commute_ratio
        subway_save_distance += od * subway_ratio * driving_distance * 2 * commute_ratio
        subway_mileage_total += od * subway_ratio * subway_mileage * 2 * commute_ratio
        for key in subway_line_distribution:
            if key in subway_distribution_dict:
                subway_distribution_dict[key] += od * subway_ratio * 2 * commute_ratio
            else:
                subway_distribution_dict[key] = od * subway_ratio * 2 * commute_ratio

    # carbon emission calculation
    fuel_emission_factor = 184  # gCO2/km
    elec_emission_factor = 78.7
    carbon = (fuel_car_distance * fuel_emission_factor + elec_car_distance * elec_emission_factor) / vehicle_occupancy / 1000

    year2info['traffic_num'] = [subway_num, bus_num, drive_num, bicycle_num]
    year2info['traffic_ratio'] = np.array(year2info['traffic_num']) / sum(year2info['traffic_num'])
    year2info['traffic_ratio'] = year2info['traffic_ratio'].tolist()
    year2info['carbon'] = carbon
    year2info['subway_distribution'] = subway_distribution_dict
    year2info['subway_mileage'] = subway_mileage_total
    year2info['fuel_car_distance'] = fuel_car_distance
    year2info['elec_car_distance'] = elec_car_distance
    return year2info, year

year2info_years = {}

num_processes = args.num_process
pool = Pool(processes=num_processes)

args_list = list(range(2010, 2024))

for result in pool.imap(yearinfo_collect_city_counterfactual, args_list):
    if result is not None:
        year2info_years[result[-1]] = result[0]
pool.close()
pool.join()

with open(basic_dir + '/years_results_overall_counterfactual_{}.json'.format(city_en_short), 'w') as f:
    json.dump(year2info_years, f)

### carbon break-even plots
construction_df = pd.read_csv('result_files/subway_construction_stats_2019.csv')
with open(basic_dir + '/years_results_overall_counterfactual_{}.json'.format(city_en_short), 'r') as f:
    results = json.load(f)
with open(basic_dir + '/years_results_overall_months_{}.json'.format(city_en_short), 'r') as f:
    results_factual = json.load(f)

carbon_save_list = {}
year_list = list(range(2010, 2021))
true_year_list = []
ratio = passenger_entry_volume_true_list[0] / passenger_entry_volume_list[0]

for year in year_list:
    signal = 0
    carbon_save_list[year] = {}
    for month in results_factual[str(year)]:
        if results[str(year)]['carbon'] - results_factual[str(year)][month]['carbon'] != 0:
            counterfactual_ratio = np.sum(results_factual[str(year)][month]['traffic_num']) / np.sum(
                results[str(year)]['traffic_num'])
            carbon_save_list[year][month] = (results[str(year)]['carbon'] * counterfactual_ratio -
                                             results_factual[str(year)][month]['carbon']) * ratio / 1e6
            if signal == 0:
                true_year_list.append(year)
                signal = 1

year2construction = {}
year2ope = {}
accu_length = 0
count = 0

start_year = 2010
accu_length = 0

for year in range(start_year, 2021):
    tmp = subway_years[subway_years['open_year'] == year]
    months = sorted(tmp['open_month'].unique())
    year2construction[year] = {}
    year2ope[year] = {}

    for month in months:
        new_subways = tmp[tmp['open_month'] == month]
        construction = 0
        for row_id in range(len(new_subways)):
            line = new_subways.iloc[row_id]['地铁线路']
            station_num, mileage = new_subways.iloc[row_id]['stationnum'], new_subways.iloc[row_id]['mileage']
            construction += construction_carbon_cal(station_num, mileage)  
            accu_length += new_subways.iloc[row_id]['mileage']
        year2construction[year][month] = construction
        year2ope[year][month] = accu_length / 1e3 *elec_intensity /365

logging.info('city counterfactual started!')
logging.info('year2construction%s' % str(year2construction))
logging.info('year2ope%s' % str(year2ope))
logging.info('carbon_save_list%s' % str(carbon_save_list))

x = []
y = []
x0 = []
z = 0

def cal(year, month):
    begin_year = int(true_year_list[0])
    year = int(year)
    month = int(month)
    return year - begin_year + (month - 1) / 12

count = 0
current_operation = 0
signal = 0
for year in true_year_list:
    last_construction_month = 0
    for month in range(1, 13):
        if month in year2construction[year]:
            signal = 1
            if month != 12:
                if current_operation != 0:
                    month_range = [months for months in carbon_save_list[year] if month in eval(months)][0]
                    x.append('{}.{}'.format(year, str(int(month) + 1)))
                    x0.append(cal(year, int(month) + 1))
                    z = z - (carbon_save_list[year][month_range] - current_operation) * (
                                int(month) - last_construction_month) * 30
                    count += 1
                    y.append(z)
                x.append('{}.{}'.format(year, str(int(month) + 1)))
                x0.append(cal(year, int(month) + 1))
                z = z + year2construction[year][month]
                y.append(z)
            else:
                if current_operation != 0:
                    month_range = [months for months in carbon_save_list[year] if int(month) in eval(months)][0]
                    x.append('{}'.format(str(int(year) + 1)))
                    x0.append(cal(str(int(year) + 1), 1))
                    z = z - (carbon_save_list[year][month_range] - current_operation) * (
                                int(month) - last_construction_month) * 30
                    count += 1
                    y.append(z)
                x.append('{}'.format(str(int(year) + 1)))
                x0.append(cal(str(int(year) + 1), 1))
                z = z + year2construction[year][month]
                y.append(z)
            current_operation = year2ope[year][month]
            last_construction_month = int(month)
            continue
        if signal == 1 and month == 12:
            month_range = [months for months in carbon_save_list[year] if 12 in eval(months)][0]
            x.append('{}'.format(str(int(year) + 1)))
            x0.append(cal(str(int(year) + 1), 1))
            count += 1
            z = z - (carbon_save_list[year][month_range] - current_operation) * (
                        int(month) - last_construction_month) * 30
            y.append(z)

plt.figure(figsize=(12, 9))
plt.rcParams['savefig.dpi'] = 100  
plt.rcParams['figure.dpi'] = 100  

plt.plot(x0, y, marker='o', linestyle='-', color='steelblue', linewidth=3)
plt.xticks(x0, x)
plt.axhline(y=0, color='indianred', linestyle='--', linewidth=3)
plt.ylabel('Carbon Debt (ton CO2)', fontsize=24)
plt.xlabel('Year', fontsize=24)
plt.yticks(fontsize=16)
plt.xticks(fontsize=6, rotation=90)
plt.savefig(basic_dir + '/carbon_debt_{}.png'.format(city_en_short), format="png", bbox_inches="tight")