from PIL import Image
import numpy as np



def simple_projection(background_image, k_x, k_y, k_z, hit_blackhole):
    resy = k_x.shape[0]
    resx = k_x.shape[1]

    pil_im = Image.open(background_image)
    width, height = pil_im.size
    pixels = np.array(pil_im)#.getdata())
    #pixels = pixels.reshape(width,height,4)
    channels = pixels.shape[2]

    if channels == 3: 
        black = np.array([ 0,0,0], dtype=np.uint8)
    else:
        black = np.array([ 0,0,0,255], dtype=np.uint8)

    def background_hit(direction):
        color = np.zeros(3)  
        if direction[2] > 1 or direction[2]<-1:
            direction = direction / np.linalg.norm(direction)
    
        #arccos loopt van [0, pi]
        theta = np.arccos(direction[2])/np.pi # [0,1]
        #arctan2 loopt van [-pi, pi]
        phi = 1/2 + 1/2*np.arctan2(direction[1], direction[0])/np.pi # [0,1]

        return pixels[int(theta*(height-1)), int((1-phi)*(width-1))]

    projection = np.zeros((resy, resx, channels))
    
    for ix, iy in np.ndindex(k_x.shape):
        if hit_blackhole[iy,ix] == 1.0:
            projection[iy, ix] = black
        else:
            projection[iy, ix] = background_hit([k_x[iy, ix], k_y[iy, ix], k_z[iy, ix]] )
    
    return Image.fromarray(projection.astype('uint8'))
