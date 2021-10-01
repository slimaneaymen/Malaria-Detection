import numpy as np
from skimage.draw import ellipse, polygon
import napari
from napari.layers import Shapes
from skimage.morphology import disk,black_tophat
from skimage.measure    import regionprops
from skimage.io         import imsave
from skimage.transform import resize
import os
import time

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
                rect, x1, x2, y1, y2 = transform_to_rect(grayscale, data, x, y)
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
    if data.shape[1] > 2:
        data = data[:,1:3]
    start = (np.min(data[:, 0]), np.min(data[:, 1]))
    end = (np.max(data[:, 0]), np.max(data[:, 1]))
    row = [start[0], start[0], end[0], end[0]]
    column = [start[1], end[1], end[1], start[1]]
    r, c = polygon(row, column, shape=(x, y))
    dim = (int(abs(end[0]-start[0])), int(abs(end[1]-start[1])))
    rect = np.zeros((max(r)-min(r),max(c)-min(c)))
    print('min(r):max(r),min(c):max(c)', min(r), max(r), min(c), max(c),'dim',dim)
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

def extract_imagette(image, labeled,raw, coords_para, coords_distrac,cells_mean=60, size1=71, size2=71, 
                     travel_output=os.getcwd()+"_output/", tophat=False, zoro=False,mask=False):
    debut=time.time()
    #existence of folder
    if not os.path.exists(travel_output):
      os.makedirs(travel_output)
    #existence of folder tophat
    if tophat:
        travel_tophat=travel_output[:-1]+'_tophat/'
        if not os.path.exists(travel_tophat):
           os.mkdir(travel_tophat)
       
    #tranforme coords_para
   
    if coords_para is False:
        pass
    else:
         if type(coords_para)==np.ndarray:
            pass
         else:
            if 2 in coords_para.shape and len(coords_para.shape)==2:
                if coords_para.shape[0]==2:
                    coords_para=coords_para.T
                    print('coords_para.shape', coords_para.shape)
            else:
                print("coords_para n'est pas à la bonne taille\ncoords_para shape=",coords_para.shape)
                return None
         list_para=list(coords_para)
    if coords_distrac is False:
        list_distrac=[]
        pass
    else:
        list_distrac =list(coords_distrac)
    
    #tophat image
    if tophat:
        black_para=black_tophat(zoro, selem=disk(5))
        black_para[black_para.mask]=0
    p=0
    pp = 0
    for region in regionprops(labeled): 
        p = p + 1
        infected=False
        distrac=False
        taille=image[region.bbox[0]:region.bbox[2],region.bbox[1]:region.bbox[3]].shape
        
        dx0 = region.bbox[0]
        dx1 = region.bbox[2]
        dy0 = region.bbox[1]
        dy1 = region.bbox[3]
        
        if(taille[0]%2!=0):
          dx1=dx1+1
        if(taille[1]%2!=0):
          dy1=dy1+1

        taille=image[dx0:dx1,dy0:dy1].shape

        if taille[0]>size1 or taille[1]>size2:
            # print("\n taille[0], [1]=",taille[0],",",taille[1])
            # print("\n size1", size1)
            z = 0
            #print('problème de taille')
        else:
          image_sortie=np.zeros([size1,size2], dtype='uint8')
          image_sortie_mask=np.zeros([size1,size2], dtype='uint8')
          image_mask = np.zeros([size1,size2], dtype='uint8')
          image_mask_one = np.ones([size1,size2], dtype='uint8')
          image_tophat=np.zeros_like(image_sortie)
          if tophat:
              image_tophat=np.zeros_like(image_sortie)
          if region.equivalent_diameter>cells_mean*2/3:
              xc,yc=region.centroid
              x=int(xc-region.bbox[0])
              y=int(yc-region.bbox[1])
              center1=int(size1/2)
              center2=int(size2/2)
              begin_x=center1-x
              begin_y=center2-y
              coords = region.coords
              #test de depassement
              if np.max(coords[:,0]-region.bbox[0]+begin_x)>size1-1:
                  recalage_x=np.max(coords[:,0]-region.bbox[0]+begin_x)-size1+1
                  begin_x=begin_x-recalage_x
              if np.max(coords[:,1]-region.bbox[1]+begin_y)>size2-1:
                  recalage_y=np.max(coords[:,1]-region.bbox[1]+begin_y)-size2+1
                  begin_y=begin_y-recalage_y
              
              diffx1 = diffx2 = int((size1 - taille[0] )/2)
              diffy1 = diffy2 = int((size2 - taille[1] )/2)

              maxx,maxy = image.shape
              
              if(dx0-diffx1<0):
                diffx2 = diffx2 + (dx0-diffx1)*(-1)
                diffx1 = diffx1 - (dx0-diffx1)*(-1)
              if(dx1+diffx2 > maxx):
                diffx1 = diffx1 + (maxx- dx1-diffx2)*(-1)
                diffx2 = diffx2 - (maxx- dx1-diffx2)*(-1)
              if(dy0-diffy1<0):
                diffy2 = diffy2 + (dy0-diffy1)*(-1)
                diffy1 = diffy1 - (dy0-diffy1)*(-1)
              if(dy1+diffy2 > maxy):
                diffy1 = diffy1 + (maxy- dy1-diffy2)*(-1)
                diffy2 = diffy2 - (maxy- dy1-diffy2)*(-1)
                
              if (mask!=True):
                image_sortie[center1-int(size1/2):center1+int(size1/2), center2-int(size2/2):center2+int(size2/2)]=\
                image[dx0-diffx1:dx1+diffx2,dy0-diffy1:dy1+diffy2]
                imsave(travel_output+ str(p)+'.tif', image[dx0-diffx1:dx1+diffx2,dy0-diffy1:dy1+diffy2])
              else:
                if (coords[:,0]-region.bbox[0]+diffx1).max() < 71 and (coords[:,1]-region.bbox[1]+diffy1).max() < 71:
                  print((coords[:,0]-region.bbox[0]+diffx1).max(),(coords[:,1]-region.bbox[1]+diffy1).max()) 
                  image_sortie_mask[coords[:,0]-region.bbox[0]+diffx1, coords[:,1]-region.bbox[1]+diffy1]=\
                  labeled[coords[:,0],coords[:,1]]
                  sh = image[dx0-diffx1:dx0-diffx1+70,dy0-diffy1:dy0-diffy1+70].shape
                  print('shape = ',sh)
                  mask_out = resize(image_sortie_mask,(sh[0],sh[0]))
                #   mask_out = (mask_out/mask_out.max()).astype(int)
                if ((dx1+diffx2) - (dx0-diffx1)) > 70 or ((dy1+diffy2) - (dy0-diffy1)) > 70:
                  image_sortie[center1-int(size1/2):center1+int(size1/2), center2-int(size2/2):center2+int(size2/2)]=\
                  image[dx0-diffx1:dx0-diffx1+70,dy0-diffy1:dy0-diffy1+70]
                  multiply = mask_out * image[dx0-diffx1:dx0-diffx1+70,dy0-diffy1:dy0-diffy1+70]
                  imsave(travel_output+ str(p)+'.tif', multiply)
                else:
                  image_sortie[center1-int(size1/2):center1+int(size1/2), center2-int(size2/2):center2+int(size2/2)]=\
                  image[dx0-diffx1:dx1+diffx2,dy0-diffy1:dy1+diffy2]
                  multiply = mask_out * image[dx0-diffx1:dx1+diffx2,dy0-diffy1:dy1+diffy2]
                  imsave(travel_output+ str(p)+'.tif', multiply)
