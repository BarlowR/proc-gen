from math import perm
from pickletools import uint8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import time
from PIL import Image

import a_star


quint_fade = lambda t : t * t * t * (t * (t * 6 - 15) + 10)
lerp = lambda t, a, b : a + t * (b - a)

permutation = [151,160,137,91,90,15,
   131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
   190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
   88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
   77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
   102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
   135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
   5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
   223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
   129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
   251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
   49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
   138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180]

p = permutation+permutation

def improved_perlin_noise(x, y, z, fade = quint_fade):

    X = int(x) &255            # FIND UNIT CUBE THAT
    Y = int(y) &255            # CONTAINS POINT.
    Z = int(z) &255
    x -= int(x)                        # FIND RELATIVE X,Y,Z
    y -= int(y)                        # OF POINT IN CUBE.
    z -= int(z)
    u = fade(x)                               # COMPUTE FADE CURVES
    v = fade(y)                               # FOR EACH OF X,Y,Z.
    w = fade(z)
    A = p[X]+Y 
    AA = p[A]+Z 
    AB = p[A+1]+Z  
    B = p[X+1]+Y 
    BA = p[B]+Z     # HASH COORDINATES OF
    BB = p[B+1]+Z   # THE 8 CUBE CORNERS,

    return lerp(w,  lerp(v, lerp(u, grad(p[AA  ], x  , y  , z   ),     # AND ADD
                                            grad(p[BA  ], x-1, y  , z   )),  # BLENDED
                                    lerp(u, grad(p[AB  ], x  , y-1, z   ),   # RESULTS
                                            grad(p[BB  ], x-1, y-1, z   ))), # FROM  8
                            lerp(v, lerp(u, grad(p[AA+1], x  , y  , z-1 ),   # CORNERS
                                            grad(p[BA+1], x-1, y  , z-1 )),  # OF CUBE
                                    lerp(u, grad(p[AB+1], x  , y-1, z-1 ),
                                            grad(p[BB+1], x-1, y-1, z-1 ))))

def grad(hash, x, y, z):
      h = hash & 15;                    # CONVERT LO 4 BITS OF HASH CODE
      if (h<8): u = x                   # INTO 12 GRADIENT DIRECTIONS.
      else: u = y     

      if (h<4): v = y
      elif (h==12 or h==14): v =  x
      else: v = z

      if ((h&1) == 0): first = u
      else: first = -u 
      
      if ((h&2) == 0): return first + v
      else: return first - v
      
def noise_map(x_dim, y_dim, scale_x, scale_y = -1, z_pos = 0, x_offset = 0, y_offset = 0, interp_func = quint_fade):
    if scale_y == -1: scale_y = scale_x
    grid = np.zeros((x_dim, y_dim))

    for i in range(y_dim):
        for j in range(x_dim):
            x_pos_world = (j+x_offset)/x_dim * scale_x
            y_pos_world = (i+y_offset)/y_dim * scale_y
            grid[i, j] = improved_perlin_noise(x_pos_world, y_pos_world, z_pos, fade=quint_fade)
    
    return grid

def rand_num_map(h):
    return np.array([[p[(p[j]+p[i]+j)%256] for j, height in enumerate(row)] for i, row in enumerate(h)])

def plot_3d(map, fig):
    x, y = np.mgrid[0:map.shape[0], 0:map.shape[1]]
    ax = fig.gca(projection='3d')   
    ax = fig.gca(projection="3d")
    ax.set_zlim3d([-1, 5])
    ax.plot_surface(x, y, map ,rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)



def world_gen(size, offset = [0,0], rock_elev = 0.75, rock_scale = .1, water_elev = 0.2, scale = 1):
    map_layers = {  2:1,
                    4:3,
                    #8:1,
                    16:2,
                    32:1,
                    64:1} #{  1:8 
                    #3:9, 
                    #5:17,
                    #10:6,
                    #20:6,
                    #30:9,
                    #90:9
                #}   
    scaling_fact = sum(map_layers.values())
    print(scaling_fact)
    for key, value in map_layers.items():
        map_layers[key] = value/scaling_fact 

    map = np.zeros(size)
    for layer, layer_scale in map_layers.items():
        if layer_scale > 0:
            map += (layer_scale * noise_map(size[0], size[1], scale_x= layer /scale, x_offset=offset[0], y_offset=offset[1]))

    delta = np.max(map)- np.min(map)
    map = (map - np.min(map))/delta

    #map[map > rock_elev+rock_scale] += (rock_scale*noise_map(size[0], size[1], scale_x=80 /scale, x_offset=offset[0], y_offset=offset[1])[map > rock_elev+rock_scale])/2 + rock_scale
    return map


def naive_convolution(map, kernel_lambda, kernel_shape = (3,3)):
    y_start = int((kernel_shape[0]-1)/2)
    x_start = int((kernel_shape[1]-1)/2)
    new_map = np.zeros((int(map.shape[0]-2*y_start), int(map.shape[1]-2*x_start)))
    for y in range (y_start, map.shape[0]-y_start):
        for x in range (x_start, map.shape[1]-x_start):
            new_map[y-y_start, x-x_start] = kernel_lambda(map[y-y_start:y+1+y_start, x-x_start:x+1+x_start])
    return(new_map)

def smoothing_kernel(sample):
    return (np.sum(sample)/np.size(sample))

def green_grad(h): #h 0-1
    return [10 + h*10, 55 + h*200, 40 + h* 30]

def blue_grad(d): #d 0-1 where 1 is surface
    d = np.sqrt(d)
    return [10 + d*10, 10 + d*30, d* 200]

def peaks(h):
    return [150+h*100]*3

def next_track_valid(new_pos, prev_pos, map):
    if not (a_star.within_bounds(new_pos, prev_pos, map)): return False

    train_high_bound = 0.7
    train_low_bound = 0.4
    max_delta = 0.2

    new_height = map[new_pos[0], new_pos[1]]
    old_height = map[prev_pos[0], prev_pos[1]]

    height_bounds = new_height > train_low_bound and new_height < train_high_bound
    height_delta = np.abs(old_height - new_height) < max_delta

    return (height_bounds and height_delta)

def build_train(map, start, end): 
    return a_star.a_star(map, start, end, passable= next_track_valid)

def train_network(map, stops):
    train_map = np.copy(map)
    stations = [[0,0]]
    network = []
    height = 0
    while (height > 0.7 or height < 0.4):
        stations[0] = [int(np.random.random() * train_map.shape[0]), int(np.random.random() * train_map.shape[1])]
        height = train_map[stations[0][0], stations[0][1]]

    i = 1
    while (i < stops):
        new_station = [0,0]
        height = 0
        while (height > 0.7 or height < 0.4):
            new_station = [int(np.random.random() * train_map.shape[0]), int(np.random.random() * train_map.shape[1])]
            height = train_map[new_station[0], new_station[1]]
        print(new_station)
        closest_stop = (stations[0], train_map.size)
        for i in range(1,len(stations)):
            dist = a_star.euclidean_distance(stations[i], new_station)
            if (dist < closest_stop[1]):
                closest_stop = (stations[i], dist)

        
        
        tracks = build_train(train_map, (closest_stop[0][0], closest_stop[0][1]), (new_station[0], new_station[1]))
        if (len(tracks) > 0):
            stations.append(new_station)
            network += tracks
            i+=1
            for track in tracks[1:-1]:
                train_map[track[0], track[1]] = 1
            print(stations)

    
    return (stations, network)


def color_builder(map, tracks = [], stations = [], rock_elev = 0.75, water_elev = 0.2):
    map = np.asarray(map)
    color_map = np.zeros((map.shape[0], map.shape[1], 3), dtype="uint8")
    green_map = green_grad((map-water_elev)/(1 - water_elev))
    color_map = np.transpose(np.asarray(green_map, dtype="uint8"), (1, 2, 0))

    blue_map = blue_grad(1+(map-water_elev)/water_elev)
    color_map[map <= water_elev] = np.transpose(np.asarray(blue_map, dtype="uint8"), (1, 2, 0))[map <= water_elev]
    #rand_map = rand_num_map(np.array(map*125+125, dtype=np.int))
    #color_map[rand_map > 250] = [90,80,40]
    color_map[np.all([map > water_elev, map < water_elev+0.02], axis=0)] = [144, 128, 78]
    color_map[map > rock_elev] = np.transpose(peaks((map[map>rock_elev] - rock_elev)/(1-rock_elev)))
    
    for track_pos in tracks:
        color_map[track_pos[0], track_pos[1]] = [60,60,60]

    for station in stations:
        color_map[station[0]-1:station[0]+1, station[1]-1:station[1]+1] = [160,160,160]
   
    return color_map


def image_from_map(map, tracks = [], stations = [], rock_elev = 0.75, water_elev = 0.2):
    colormap = color_builder(map, tracks = tracks, stations = stations,rock_elev = rock_elev, water_elev = water_elev)
    img = Image.fromarray(colormap)
    return img.resize([1024]*2, Image.NEAREST)

if __name__ == "__main__":
    np.random.seed(12345)
    size = ([1024]*2)
    rock_elev = 0.75
    water_elev = .35

    map = world_gen(size, rock_elev = rock_elev, water_elev = water_elev, scale=.5)
    map = naive_convolution(map, smoothing_kernel)  
    map = naive_convolution(map, smoothing_kernel)     

    
    stations, tracks = train_network(map, 5)
    print(stations)
    image_from_map(map, tracks = tracks, stations = stations, rock_elev = rock_elev, water_elev = water_elev).show()
    #plot_3d(map, plt.figure())
    #plt.imshow(map)
    #plt.show()
