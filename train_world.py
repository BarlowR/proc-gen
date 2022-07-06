import numpy as np
import world_creation.bitmap_world as bmw
import world_creation.trains.train_builder as tb


if __name__ == "__main__":
    np.random.seed(150)
    size = ([600, 300])
    rock_elev = 0.75
    water_elev = .35

    map = bmw.world_gen(size, rock_elev = rock_elev, water_elev = water_elev, scale=.6)
    map = bmw.naive_convolution(map, bmw.smoothing_kernel)  
    map = bmw.naive_convolution(map, bmw.smoothing_kernel)     

    
    stations, tracks = tb.train_network(map, 8)
    print(stations)
    img = bmw.image_from_map(map, tracks = tracks, stations = stations, rock_elev = rock_elev, water_elev = water_elev)
    img.save("map.png")
    img.show()
    #plot_3d(map, plt.figure())
    #plt.imshow(map)
    #plt.show()
