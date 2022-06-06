from math import perm
from pickletools import uint8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

from noise import perlin_noise as perlin_noise
from world_creation.trains import train_builder as train_builder

def rand_proc_num_map(h):
    return np.array([[p[(p[j]+p[i]+j)%256] for j, height in enumerate(row)] for i, row in enumerate(h)])

def plot_3d(map, fig):
    x, y = np.mgrid[0:map.shape[0], 0:map.shape[1]]
    ax = fig.gca(projection='3d')   
    ax = fig.gca(projection="3d")
    ax.set_zlim3d([-1, 5])
    ax.plot_surface(x, y, map ,rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)


#generate a world map with given parameters
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
            map += (layer_scale * perlin_noise.noise_map(size[0], size[1], scale_x= layer /scale, x_offset=offset[0], y_offset=offset[1]))

    delta = np.max(map)- np.min(map)
    map = (map - np.min(map))/delta

    #map[map > rock_elev+rock_scale] += (rock_scale*noise_map(size[0], size[1], scale_x=80 /scale, x_offset=offset[0], y_offset=offset[1])[map > rock_elev+rock_scale])/2 + rock_scale
    return map


# a naive convolutional kernal operation implementation
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

# gradient lambda
def green_grad(h): #h 0-1
    return [10 + h*10, 55 + h*200, 40 + h* 30]

# gradient lambda
def blue_grad(d): #d 0-1 where 1 is surface
    d = np.sqrt(d)
    return [10 + d*10, 10 + d*30, d* 200]

# gradient lambda
def peaks(h):
    return [150+h*100]

#build a color map from a height map, a list of tracks & stations, and rock/water elevations
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
        color_map[station[0]-1:station[0]+2, station[1]-1:station[1]+2] = [160,160,160]
   
    return color_map


def image_from_map(map, tracks = [], stations = [], rock_elev = 0.75, water_elev = 0.2):
    colormap = color_builder(map, tracks = tracks, stations = stations,rock_elev = rock_elev, water_elev = water_elev)
    img = Image.fromarray(colormap).transpose(Image.TRANSPOSE)
    return img




if __name__ == "__main__":
    np.random.seed(1250)
    size = ([1200, 600])
    rock_elev = 0.75
    water_elev = .35

    map = world_gen(size, rock_elev = rock_elev, water_elev = water_elev, scale=.3)
    map = naive_convolution(map, smoothing_kernel)  
    map = naive_convolution(map, smoothing_kernel)     

    
    stations, tracks = train_builder.train_network(map, 20)
    print(stations)
    img = image_from_map(map, tracks = tracks, stations = stations, rock_elev = rock_elev, water_elev = water_elev)
    img.save("map.png")
    img.show()
    #plot_3d(map, plt.figure())
    #plt.imshow(map)
    #plt.show()
np.math.pi/2