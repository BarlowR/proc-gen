

#takes desired x,y,z positions on isometric grid and returns 2d x and y projection tuple
def xyz_to_iso_xy(x,y,z):
    iso_x = (int)(0.5*x - 0.5*y)
    iso_y = (int)(0.25*x + 0.25*y + z)

    return (iso_x, iso_y)
