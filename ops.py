import numpy as np 
import cv2 

def get_gaussian_filter(shape:tuple[int, int], sigma:float=0.5) -> np.ndarray:
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def gaussianblur_custom(image:np.ndarray, kernel_size:int, sigma:float=0.5) -> np.ndarray:
    kernel = get_gaussian_filter((kernel_size,kernel_size)) 
    blurred_img = cv2.filter2D(image, ddepth=-1, kernel=kernel)

    return blurred_img