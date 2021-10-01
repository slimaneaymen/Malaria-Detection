import numpy as np
from skimage.draw import ellipse, polygon
import napari
from napari.layers import Shapes

def polygon_shape(data, x, y):
    """This function generate coordinates of pixels within a polygon.
    Parameters
    ----------
    data : numpy array
        Array containing the coordinates of the N vertices in D dimensions 
        that make up the polygon shape. 
    x : int
        Image.shape[0]. 
    y : int
        Image.shape[1].
        Image shape used to determine the maximum bounds of the output coordinates. 
        This is useful for clipping polygon that exceed the image size.
    Returns
    -------
    array of int
        The coordinates of all pixels in the polygon.    
    """
    vertice_row = data[:, 0]
    vertice_column = data[:, 1]
    r, c = polygon(vertice_row, vertice_column, shape=(x, y))
    return r, c


def ellipse_shape(data, x, y):
    """This function generate coordinates of pixels within an ellipse.
    Parameters
    ----------
    data : numpy array
        Array containing the coordinates of the N vertices in D dimensions 
        that make up the ellipse shape. 
    x : int
        Image.shape[0]. 
    y : int
        Image.shape[1].
        Image shape used to determine the maximum bounds of the output coordinates. 
        This is useful for clipping ellipse that exceed the image size.
    Returns
    -------
    array of int
        The coordinates of all pixels in the ellipse.    
    """
    center_row, center_column = int(np.min(data[:, 0])) + int((np.max(data[:, 0]) - np.min(
        data[:, 0]))/2), int(np.min(data[:, 1])) + int((np.max(data[:, 1]) - np.min(data[:, 1]))/2)
    minor_axe, major_axe = int((np.max(
        data[:, 0]) - np.min(data[:, 0]))/2), int((np.max(data[:, 1]) - np.min(data[:, 1]))/2)
    r, c = ellipse(center_row+1, center_column+1,
                   minor_axe, major_axe, shape=(x, y))
    return r, c

def patch_extraction(viewer, img_name, sh_name):        
    """This function is for extracting data from patches.
    Parameters
    ----------
    viewer : object 
        napari.Viewer object that includes all the data added to napari.  
    img_name : str
        name of the selected Image layer. 
    sh_name : str
        name of the selected Shape layer. 

    Returns
    -------
    numpy array
        Array of the same size as the input image containing only the patches data.    
    """

    shape_layer = [{'types': layer_.shape_type, 'data': layer_.data}
                    for layer_ in viewer.layers if isinstance(layer_, Shapes) and layer_.name == sh_name]
    grayscale = get_image(img_name,viewer)
    x, y = grayscale.shape
    output =np.zeros((x,y)) 
    if len(shape_layer) == 0:
        output = grayscale
    else:    
        for i in range(len(shape_layer)):
            for j, type_ in enumerate(shape_layer[i]['types']):
                data = shape_layer[i]['data'][j]
                rect, x1, x2, y1, y2 = transform_to_rect(grayscale, data.astype(int), x, y)
                output[x1:x2,y1:y2] = rect
    return output 
                   

def transform_to_rect(image, data, x, y):
    """This function generate a rectangle container of any shape.
    Parameters
    ----------
    image : numpy array
        Array containing the image data. 
    data : numpy array
        Array containing the coordinates of the N vertices in D dimensions 
        that make up the rectangle shape. 
    x : int
        Image.shape[0]. 
    y : int
        Image.shape[1].
        Image shape used to determine the maximum bounds of the output coordinates. 
    Returns
    -------
    array of int
        The coordinates of all pixels in the rectangle.    
    """

    start = (np.min(data[:, 0]), np.min(data[:, 1]))
    end = (np.max(data[:, 0]), np.max(data[:, 1]))
    row = [start[0], start[0], end[0], end[0]]
    column = [start[1], end[1], end[1], start[1]]
    r, c = polygon(row, column, shape=(x, y))
    dim = (abs(end[0]-start[0]), abs(end[1]-start[1]))
    rect = np.zeros((dim))
    # print('min(r):max(r),min(c):max(c)', min(r), max(r), min(c), max(c))
    rect[:, :] = image[min(r):max(r), min(c):max(c)]
    return rect,min(r),max(r), min(c),max(c)

def get_image(name,viewer):
    """This function is for getting the selected image.
    Parameters
    ----------
    name : str
        name of the selected image layer. 
    viewer : object 
        napari.Viewer object that includes all the data added to napari.  

    Returns
    -------
    numpy array
        Array containing the selected image data.    
    """

    image_layer = [{'name': layer_.name, 'shape': layer_.data.shape, 'data': layer_.data}
        for layer_ in viewer.layers if isinstance(layer_, napari.layers.Image) and layer_.name == name]
    image = image_layer[0]['data']
    return image
