import path_planning.a_star as a_star
import numpy as np


#check if the next train track position is valid
#new_pos : (x,y)
#prev_pos : (x,y)
#map: height map in (x,y) coordinates
# train_high_bound  high elevation bound for train tracks
# train_low_bound = lower elevation bound for train tracks
# max_delta : maximum elevation change between steps
# returns validity
def next_track_valid(new_pos, prev_pos, map, train_high_bound = 0.7, train_low_bound = 0.4, max_delta = 0.2):

    #check if its on the map
    if not (a_star.within_bounds(new_pos, prev_pos, map)): return False

    #grab elevations of n 
    new_height = map[new_pos[0], new_pos[1]]
    old_height = map[prev_pos[0], prev_pos[1]]

    #check bounds and elevation change
    height_bounds = new_height > train_low_bound and new_height < train_high_bound
    height_delta = np.abs(old_height - new_height) < max_delta

    return (height_bounds and height_delta)

# build train tracks from start to end on map with a_star
# map: height map in (x,y) coordinates
# start: (x,y)
# end: (x,y)
# returns a list of train track coordinates
def build_train(map, start, end): 
    return a_star.a_star(map, start, end, passable= next_track_valid)


# build a network of train tracks
# map: elevatiin map
# stops: desired train stops
# return: (stations, rail_network)
#   stations: (x,y)
#   rail_network: list of track lists
def train_network(map, stops):

    station_bounds = [0.4, 0.6]
    train_map = np.copy(map)
    stations = []
    rail_network = []
    station_links = {} #index in station : index in stations
    height = 0

    i = 0

    new_station = [0,0]
    height = 0
    while (height < station_bounds[0] or height > station_bounds[1]):
        new_station = [int(np.random.random() * train_map.shape[0]), int(np.random.random() * train_map.shape[1])]
        height = train_map[new_station[0], new_station[1]]
    
    stations.append(new_station)

    while(i < stops-1):

        # find a new station spot
        new_station = [0,0]
        height = 0
        while (height < station_bounds[0] or height > station_bounds[1]):
            new_station = [int(np.random.random() * train_map.shape[0]), int(np.random.random() * train_map.shape[1])]
            height = train_map[new_station[0], new_station[1]]
        print(new_station)

        connected = False
        #try to connect it to existing stations
        for i in range(len(stations)):
            tracks = build_train(train_map, (stations[i][0], stations[i][1]), (new_station[0], new_station[1]))
            if (len(tracks) > 0 and len(tracks) < 500): #terribly inefficient, smh
                connected = True
                new_idx = len(stations)
                station_links[new_idx] = [i]
                if (i in station_links.keys()): station_links[i].append(new_idx)
                else: station_links[i] = [new_idx]
                rail_network += tracks
                for track in tracks[2:-2]:
                    train_map[track[0]-1:track[0]+1, track[1]-1:track[1]+1] = 1

        if (connected):
            stations.append(new_station)
            i += 1
        print(station_links)

    
    return (stations, rail_network)