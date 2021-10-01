import numpy as np
import os
from skimage.filters.thresholding import threshold_otsu,threshold_local
from scipy.ndimage.morphology import binary_fill_holes, binary_closing
from skimage.morphology import remove_small_objects
from skimage.measure import label
from skimage.segmentation import clear_border
from PIL import Image as Im
import ntpath
from pathlib import Path
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.util import img_as_ubyte
from tensorflow.keras.models import model_from_json
import cv2
from napari.layers import Shapes
from skimage.segmentation import join_segmentations
from .shapes_ import ellipse_shape, polygon_shape, patch_extraction, get_image
from napari.layers import Shapes
from cellpose.models import CellposeModel



def save_as_tiff(viewer, path_):
    """ This function is responsible for saving the selected layers  as '.tiff' images 

    Parameters
    ----------
    viewer : object 
        napari.Viewer object that includes all the data added to napari.  
    path : str
        the path where to save the layers.  
    """
    selected_layers = list(viewer.layers.selection)
    path = str(path_)
    from skimage.io import imsave
    for i in range(len(selected_layers)):
        data = selected_layers[i].data
        path = str(path)+'/'+str(selected_layers[i].name)
        imsave(path + ".tiff",data)
        path = str(path_)

def load_images(viewer, path_, choice):
    """ This function is responsible for loading images as Labels or Image layers

    Parameters
    ----------
    viewer : object 
        napari.Viewer object that includes all the data added to napari.  
    path : str
        the path where to save the layers.  
    """
    from skimage.io import imread
    for l in path_[0]:
        print(l,type(l))
        image = imread(str(l))
        basename_ = ntpath.basename(l)
        str_ = basename_.split('.')
        name = ''.join(str_[0:len(str_)-1])
        if choice == 'label':
            viewer.add_labels(image, name=name)
        else:
            viewer.add_image(image, name=name)

# Classic segmentation algorithms

def segment_image(image, threshold, block_size):
    """This function makes a classic image segmentation using different types of thresholds.
    Parameters
    ----------
    image : numpy array
        Image channel. 
    threshold : float
        Used in case of manual_thresholding, s.t 0<threshold<255. 
    block_size : int
        The characteristic size surrounding each pixel, that defines a local neighborhoods
        on which a local or dynamic thresholding is calculated. 

    Returns
    -------
    numpy array
                Segmented and labeled images. 
    """
    global_thresh = threshold_otsu(image)
    if threshold == 0:
        binary = image > global_thresh
    elif threshold == 1:
        local_thresh = threshold_local(image, block_size, offset=0)
        binary = image > local_thresh
    else:
        binary = image > threshold
    # remove small dark spots (i.e. “pepper”) and connect small bright cracks.
    amask = binary_fill_holes(binary_closing(binary))
    # remove artifacts connected to image border
    cleared = remove_small_objects(clear_border(amask), 20)
    # label image regions
    labels = label(cleared)
    return global_thresh, labels, cleared

class AnnotatorShapes(Shapes):
    def __init__(self):
        super().__init__()

    def on_click(self, viewer, threshold, block_size, img_name, sh_name):
        """ Global function

        This function is responsible for extracting image and shapes data from napari viewer's layers 
        and do the processing (segmentation, labelling) then add the Segmented and labeled images to labels layer.
        Parameters
        ----------
        viewer : object 
            napari.Viewer object that includes all the data added to napari.  
        threshold : int
            Used in case of simple_threshold type, s.t 0<threshold<255. 
        block_size : int
            The characteristic size surrounding each pixel, that defines a local neighborhoods
            on which a local or dynamic thresholding is calculated. 
        img_name : str
            name of the selected Image layer. 
        sh_name : str
            name of the selected Shapes layer. 

        """
        grayscale = get_image(img_name,viewer)
        shape_layer = [{'types': layer_.shape_type, 'data': layer_.data}
                    for layer_ in viewer.layers if isinstance(layer_, Shapes) and layer_.name == sh_name]

        x, y = grayscale.shape
        labeled_image = np.zeros((x, y), dtype=np.uint8)
        cleared_image = np.zeros((x, y), dtype=np.uint8)
        if grayscale.dtype =='uint16':
            grayscale = img_as_ubyte(grayscale)        
        if len(shape_layer) == 0:
            Otsu_thresh, labels, cleared = segment_image(
                    grayscale, threshold, block_size)
            labeled_image = labels
            cleared_image = cleared 
        else:    
            for i in range(len(shape_layer)):
                for j, type_ in enumerate(shape_layer[i]['types']):
                    data = shape_layer[i]['data'][j]
                    Otsu_thresh, labels, cleared = segment_image(
                        grayscale, threshold, block_size)
                    if type_ == 'ellipse':
                        r, c = ellipse_shape(data, x, y)
                        labeled_image[r, c] = labels[r, c]
                        cleared_image[r, c] = cleared[r, c]
                    if type_ == 'polygon' or type_ == 'path' or type_ == 'rectangle':
                        r, c = polygon_shape(data.astype(int), x, y)
                        labeled_image[r, c] = labels[r, c]
                        cleared_image[r, c] = cleared[r, c]
                    # if transform == 'Yes':
                    #     rect = transform_to_rect(grayscale, data.astype(int), x, y)
                    #     viewer.add_image(rect, name = 'rectangular_shape_' + str(i))
        labels_layer = viewer.add_labels(labeled_image, name='labels')
        labels_layer = viewer.add_labels(cleared_image, name='mask')
        return Otsu_thresh
# Advanced segmentation algorithms

def stardist_segmentation(viewer, img_name, sh_name, type):
    """This function does image segmentation using StarDist algorithm.
    Parameters
    ----------
    viewer : object 
        napari.Viewer object that includes all the data added to napari.  
    img_name : str
        name of the selected Image layer. 
    sh_name : str
        name of the selected Shapes layer. 
    type: str
        '2D_versatile_fluo' & '2D_paper_dsb2018': Versatile (fluorescent nuclei) and DSB 2018 
                                                  (from StarDist 2D paper) that were both trained 
                                                  on a subset of the DSB 2018 nuclei segmentation 
                                                  challenge dataset.
        '2D_versatile_he': Versatile (H&E nuclei) that was trained on images from the MoNuSeg 2018 
                           training data and the TNBC dataset from Naylor et al. (2018).                                               
    """

    model = StarDist2D.from_pretrained(type)    
    rect = patch_extraction(viewer, img_name, sh_name)
    if rect.dtype == 'uint16':
        rect = img_as_ubyte(rect)            
    labels_, _ = model.predict_instances(normalize(rect))                
    viewer.add_labels(labels_, name='StarDist_Label')
    viewer.add_labels(labels_ > 0, name='StarDist_mask')
      
def cellpose_segmentation(viewer, img_name, sh_name,param):
    """This function does image segmentation using Cellpose algorithm.
    Parameters
    ----------
    viewer : object 
        napari.Viewer object that includes all the data added to napari.  
    img_name : str
        name of the selected Image layer. 
    sh_name : str
        name of the selected Shapes layer. 
    param: dict
        A dictionary that contain a set of parameters for Cellpose model and for the evaluation of the model    
    """
    rect = patch_extraction(viewer, img_name, sh_name)
    viewer.add_image(rect, name='rect')
    print('param["model_type"]',param)
    model = CellposeModel(gpu=param['GPU'], pretrained_model= param['pretrained_pkl'], 
            model_type=param["model_type"], torch=True, diam_mean=param["diam_mean"], net_avg=True, 
            device=None,residual_on=True, style_on=True, concatenation=False, nchan=2)        
    masks,flows,styles = model.eval(rect, batch_size=1, channels=param["channels"], channel_axis=None, 
                                    z_axis=None, normalize=True, invert=False, rescale=None, 
                                    diameter=param["diam_mean"], do_3D=False, anisotropy=None, 
                                    net_avg=True, augment=False, tile=True, tile_overlap=0.1, 
                                    resample=True, interp=True, flow_threshold=param["flow_threshold"], 
                                    cellprob_threshold=param["cellprob_threshold"], compute_masks=True, min_size=param["min_size"], 
                                    stitch_threshold=0.0, progress=None)
    viewer.add_image(flows[0], name = 'flow')
    viewer.add_labels(masks, name = 'mask')

def Unet_segmentation(viewer, img_name,sh_name):
    """This function does image segmentation using pretrained Unet algorithm.
    Parameters
    ----------
    viewer : object 
        napari.Viewer object that includes all the data added to napari.  
    img_name : str
        name of the selected Image layer. 
    sh_name : str
        name of the selected Shapes layer. 
    """

    directory = Path(os.path.dirname(os.path.abspath(__file__)))
    model_path = str(directory.parent.absolute()) + '/pretrained models/Unet_contour_assist/models/model_real.json'
    weight_path = str(directory.parent.absolute()) + '/pretrained models/Unet_contour_assist/models/model_real_weights.h5'
    # load json and create model    
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weight_path)
    print("Loaded model from disk")
    im = patch_extraction(viewer, img_name, sh_name)
    if im.dtype == 'uint16':
        im = img_as_ubyte(im)        
    img = np.array(Im.fromarray(im).resize((256,256)))
    tmp = np.zeros((1,256,256,3))
    tmp[0,:,:,0] = tmp[0,:,:,1] = tmp[0,:,:,2] = img  
    label_= loaded_model.predict(tmp)
    label_1 = label_.reshape((256,256))
    th = threshold_otsu(label_1)
    mask = label_1 < th
    mask = np.array(Im.fromarray(mask).resize((im.shape[1],im.shape[0])))
    viewer.add_labels(mask, name='Unet_mask')
    viewer.add_labels(label(mask), name='Unet_labels')    
    
# Region Growing segmentation algorithm

class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x
    def getY(self):
        return self.y

def getGrayDiff(img,currentPoint,tmpPoint):
    """This function does gray difference between two points.
    Parameters
    ----------
    img : numpy array
        Array containing the image data. 
    currentPoint : numpy array
        Array containing the position of one seed point. 
    tmpPoint : numpy array
        Array containing the position of one of the 8 neighboring points to the seed point. 
    """

    return abs(int(img[currentPoint.x,currentPoint.y]) - int(img[tmpPoint.x,tmpPoint.y]))

def selectConnects(p):
    """This function returns the position of the neighboring points to the seed.
    Parameters
    ----------
    p : int
        p==0: this returns 4 neighboring points
        p!=0: this returns 8 neighboring points

    Returns
    -------
    numpy array
        Array containing the neighboring points.    
    """
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
            Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [ Point(0, -1), Point(1, 0),Point(0, 1), Point(-1, 0)]
    return connects

def regionGrow(img,seeds,thresh,p = 1):
    """This function does an image segmentation using RegionGrowing algorithm.

    Parameters
    ----------
    img : numpy array
        Array containing the image data. 
    seeds : numpy array     
        Array containing the positions of the seed points. 
    thresh: int
        Threshold defining the gray difference tolerated between a seed and the neighboring pixels.
    p : int
        p==0: this returns 4 neighboring points
        p!=0: this returns 8 neighboring points
    Returns
    -------
    numpy array
        Array a labeled image where 1 is foreground and 0 is background.    
    """

    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)
    while(len(seedList)>0):
        currentPoint = seedList.pop(0)
        seedMark[currentPoint.x,currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
            if grayDiff < thresh and seedMark[tmpX,tmpY] == 0:
                seedMark[tmpX,tmpY] = label
                seedList.append(Point(tmpX,tmpY))
    return seedMark

def region_growing(viewer, image_layer,thresh):
    """This function does an image segmentation using RegionGrowing and returns the mask and the contour for each mask.

    Parameters
    ----------
    viewer : object 
        napari.Viewer object that includes all the data added to napari.  
    image_layer : object 
        napari.layers.Image object that includes the selected Image layer. 
    thresh: int
        Threshold defining the gray difference tolerated between a seed and the neighboring pixels.
    """    
    im = image_layer.data
    points = viewer.layers['Points'].data    
    seeds = []
    for j in range(points.shape[0]):
        seeds.append(Point(round(points[j,0]),round(points[j,1])))
    binaryImg = regionGrow(im,seeds,thresh)
    binaryImg = binary_closing(binaryImg)
    binaryImg = binary_fill_holes(binaryImg)    
    
    tmp = np.zeros((im.shape))
    contours, hierarchy = cv2.findContours(np.uint8(binaryImg),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    for j in range(len(contours)):
        c = contours[j].reshape((contours[j].shape[0],contours[j].shape[2]))
        for i in range(c.shape[0]):
            tmp[c[i,1],c[i,0]] = 2
    viewer.add_labels(tmp.astype(int), name='contour')
    viewer.add_labels(binaryImg.astype(int), name='mask')


def assemble_labels (viewer,raw_label,mod_label):
    """This function does an assemblage of labels in one label image.

    Parameters
    ----------
    viewer : object 
        napari.Viewer object that includes all the data added to napari.  
    raw_label : object 
        napari.layers.Labels object that includes the origin label layer. 
    mod_label : object 
        napari.layers.Image object that includes the label layer to assemble with the origin one. 
    """    
    tmp = binary_fill_holes(mod_label)
    thresh = 1
    points = viewer.layers['Points'].data    
    im = raw_label
    seeds = []
    for j in range(points.shape[0]):
        seeds.append(Point(round(points[j,0]),round(points[j,1])))
    binaryImg = regionGrow(im,seeds,thresh)
    inverse = 1-binaryImg.astype(int)
    multiplied = cv2.multiply(inverse, im,dtype=cv2.CV_8U)
    merged = join_segmentations(multiplied.astype(int),tmp.astype(int))
    viewer.add_labels(merged>0, name='merged_mask')
    viewer.add_labels(label(merged.astype(int)), name='merged_labels')   
            
