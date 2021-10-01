import os
from skimage.filters.thresholding import threshold_otsu,threshold_local
from scipy.ndimage.morphology import binary_fill_holes, binary_closing
from skimage.morphology import remove_small_objects
from skimage.measure import label,regionprops
from skimage.segmentation import join_segmentations, clear_border
from skimage.util import img_as_ubyte
from skimage import img_as_uint
from skimage.io import imread, imsave
import numpy as np
from PIL import Image as Im
import ntpath
from pathlib import Path
from matplotlib import pyplot as plt
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from tensorflow.keras.models import model_from_json
import cv2
from napari.layers import Shapes
from .shapes_ import ellipse_shape, polygon_shape, patch_extraction, get_image
from napari.layers import Shapes
from cellpose.models import CellposeModel
from .fonction_compteur import *
from .fonction_compteur_datagenerator import *
from .fonction_compteur_segmentation import *
from .fonction_compteur_affiche import *


def extract_cells(image, labeled,raw, coords_para, coords_distrac,cells_mean=60, size1=71, size2=71, 
                     travel_output=os.getcwd()+"_output/", tophat=False, zoro=False,mask=False):
    debut=time.time()
    #existence of folder
    if not os.path.exists(travel_output):
        os.makedirs(travel_output)
    travel_output1 = travel_output+'out/'
    if not os.path.exists(travel_output1):
        os.makedirs(travel_output1)     
        #%mkdir -p travel_output
        #tf.compat.v1.gfile.MkDir(travel_output)
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
                    if ((dx1+diffx2) - (dx0-diffx1)) > 70 or ((dy1+diffy2) - (dy0-diffy1)) > 70:
                        image_sortie[center1-int(size1/2):center1+int(size1/2), center2-int(size2/2):center2+int(size2/2)]=\
                        image[dx0-diffx1:dx0-diffx1+70,dy0-diffy1:dy0-diffy1+70]
                        multiply = image[dx0-diffx1:dx0-diffx1+70,dy0-diffy1:dy0-diffy1+70]
                    else:
                        image_sortie[center1-int(size1/2):center1+int(size1/2), center2-int(size2/2):center2+int(size2/2)]=\
                        image[dx0-diffx1:dx1+diffx2,dy0-diffy1:dy1+diffy2]
                        multiply = image[dx0-diffx1:dx1+diffx2,dy0-diffy1:dy1+diffy2]
                else:
                    if (coords[:,0]-region.bbox[0]+diffx1).max() < 71 and (coords[:,1]-region.bbox[1]+diffy1).max() < 71:
                        print((coords[:,0]-region.bbox[0]+diffx1).max(),(coords[:,1]-region.bbox[1]+diffy1).max()) 
                        image_sortie_mask[coords[:,0]-region.bbox[0]+diffx1, coords[:,1]-region.bbox[1]+diffy1]=\
                        labeled[coords[:,0],coords[:,1]]
                        from skimage.transform import resize
                        mask_out = resize(image_sortie_mask,(70,70))
                        # mask_out = (mask_out/mask_out.max()).astype(int)
                    if ((dx1+diffx2) - (dx0-diffx1)) > 70 or ((dy1+diffy2) - (dy0-diffy1)) > 70:
                        image_sortie[center1-int(size1/2):center1+int(size1/2), center2-int(size2/2):center2+int(size2/2)]=\
                        image[dx0-diffx1:dx0-diffx1+70,dy0-diffy1:dy0-diffy1+70]
                        multiply = mask_out * image[dx0-diffx1:dx0-diffx1+70,dy0-diffy1:dy0-diffy1+70]
                    else:
                        image_sortie[center1-int(size1/2):center1+int(size1/2), center2-int(size2/2):center2+int(size2/2)]=\
                        image[dx0-diffx1:dx1+diffx2,dy0-diffy1:dy1+diffy2]
                        multiply = mask_out * image[dx0-diffx1:dx1+diffx2,dy0-diffy1:dy1+diffy2]
                # imsave(travel_output+ str(p)+'.tif', np.uint(multiply))
                imsave(travel_output1+ str(p)+'.tif', multiply)

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
    if image.dtype =='uint16' or image.dtype =='uint64':
        image = (image/image.max())*255        
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

        if grayscale.dtype =='uint16'or grayscale.dtype =='uint64':
            grayscale = (grayscale/grayscale.max())*255        
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


def hough_segmentation(viewer, img_name, sh_name,param):
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
    # viewer.add_image(rect,name = 'patch_extracted')
    # image_name = 'G/9994A323901.tif'
    # image = plt.imread(image_name)
    image_mask = decoupe_mask(rect, verbose='False')
    # viewer.add_labels(image_mask, name='image_mask1')    
    zoro=np.ma.masked_where(image_mask==0,image_mask)
    viewer.add_labels(image_mask.astype(int)>image_mask.astype(int).min(), name='image_mask')
    labeled_list=Hough_by_thres(rect, zoro, cells_mean=param['cell_mean'], verbose=param['verbose'],condition=['seuil',1,0.9,1], 
                    edges=None,labeled_list=[], exemple=[False])   
    labeled_list=recon_image(labeled_list,verbose=param['verbose'])
    labeled=labeled_list[0] 
    #test if region is too small ot too big
    classe, amas_cells=test_region(labeled, cells_mean=param['cell_mean'], threshold='hough_iter', bord=True)
    result=affiche(rect, labeled,classe, title=" premier filtrage",boxes=["ellipse","blue"])
    save=result[1]
    save=np.array(save*255, dtype='uint8')
    save=draw_ellipse_perso(save, classe)
    save=draw_ellipse_perso(save, amas_cells)
    labeled_conv=complet_cells(classe+amas_cells, labeled, verbose='True')
    classe=regionprops(labeled_conv)
    labeled_conv = img_as_uint(labeled_conv)
    imsave(param['out_dir']+ '/' +img_name+"_labeling.png", labeled_conv)
    shape_layer = [{'types': layer_.shape_type, 'data': layer_.data}
                    for layer_ in viewer.layers if isinstance(layer_, Shapes) and layer_.name == sh_name]
    x, y = labeled_conv.shape
    output = np.empty((x,y))
    if len(shape_layer) == 0:
        output = labeled_conv
    else:    
        for i in range(len(shape_layer)):
            for j, type_ in enumerate(shape_layer[i]['types']):
                data = shape_layer[i]['data'][j]
                rect1, x1, x2, y1, y2 = transform_to_rect(labeled_conv, data, x, y)
                output[x1:x2,y1:y2] = rect1
    viewer.add_labels(output.astype(int), name='labels')

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
    print('model',model)
    rect = patch_extraction(viewer, img_name, sh_name)
    print('rect',rect,'rect.dtype',rect.dtype)
    if rect.dtype == 'uint16':
        rect = img_as_ubyte(rect)            
    labels_, _ = model.predict_instances(normalize(rect))                
    print('labels_',labels_)
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

    directory = Path(os.path.dirname(os.path.abspath('_segmentation.py')))
    print(str(directory))
    # model_path = str(directory.parent.absolute()) + '/pretrained models/Unet_contour_assist/models/model_real.json'
    model_path = str(directory) + '/pretrained models/Unet_contour_assist/models/model_real.json'
    weight_path = str(directory) + '/pretrained models/Unet_contour_assist/models/model_real_weights.h5'
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
            