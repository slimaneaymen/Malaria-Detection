"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""


import sys
from os import path 
from operator import invert
from PyQt5.QtWidgets import QMainWindow
sys.path.append(
    '/home/slimane/Desktop/Big_Annotator/bigannotator/NEW/bigannotator/pretrained models')
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import *
from typing import TYPE_CHECKING
# from principle_function import global_segmentation
from magicgui.widgets import LineEdit
from napari.components.layerlist import LayerList
from ._segmentation import *
from .shapes_ import *
from .image_reconstraction import *
from .image_classification import *
from napari import Viewer                                     
from napari.types import LabelsData, ImageData
import napari
import numpy as np
import pathlib
from magicgui import magic_factory
from PyQt5.QtCore import QDir
from napari_plugin_engine import napari_hook_implementation
import cv2
from skimage.transform import resize
from skimage.filters import gaussian,difference_of_gaussians
from PIL import Image as Im
from skimage.feature import blob_log
from skimage.draw import circle_perimeter
from napari.layers import Shapes
from ast import literal_eval
from math import pi as pi
from .function_model import create_model
from .my_classes import DataGeneratorPhase
import scipy.io.matlab as mio
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers  import Adam
from cellpose.models import CellposeModel
import glob
from imageio import *
from sklearn.model_selection import train_test_split
import tifffile
from cellpose.dynamics import labels_to_flows


def get_image(name,viewer,ch):
    """This function is for getting the selected image.
    Parameters
    ----------
    name : str
        name of the selected image layer. 
    viewer : object 
        napari.Viewer object that includes all the data added to napari.  
    ch : int
        The channel number. 

    Returns
    -------
    numpy array
        Array containing the selected image data.    
    """
    
    image_layer = [{'name': layer_.name, 'shape': layer_.data.shape, 'data': layer_.data}
        for layer_ in viewer.layers if isinstance(layer_, napari.layers.Image) and layer_.name == name]

    x, y = image_layer[0]['shape'][:2]        
    if ch == 'gray':
        grayscale = image_layer[0]['data']
        gray = (Im.fromarray(grayscale)).convert('L')
        grayscale = np.array(gray).reshape(x, y)
    elif ch =='None':
        grayscale = image_layer[0]['data']    
    else:
        print(len(image_layer[0]['shape']),image_layer[0]['shape'][0])
        if len(image_layer[0]['shape']) > 2:
            print('image_layer[0][shape][0]',image_layer[0]['shape'][0])
            if image_layer[0]['shape'][0] >36:
                grayscale = image_layer[0]['data'][:, :, int(ch)]
            else :
                grayscale = image_layer[0]['data'][int(ch), :, :]   
            # grayscale = img_as_ubyte(grayscale)                
        else:
            grayscale = image_layer[0]['data']
            # grayscale = img_as_ubyte(grayscale)
    return grayscale

class ImagePreprocessing (QMainWindow):   
     
    """ This class widget allows the user to do image preprocessing.

    This widget gives the user the ability to:
    - Choose a channel.
    - Smooth an image by selecting a sigma value.
    - Pass_band filter an image by selecting a min and max sigma values.
    - Invert an image
    - Crop different regions of the image. 

    """
    def __init__(self, napari_viewer, parent=None):
        """ QWidget.__init__ method.

        Parameters
        ----------
        napari_viewer : instance
            Access to napari viewer in order to add widgets.
        parent : class
            Parent class (QWidget)
        """

        super().__init__(parent)
        
        self.viewer = napari_viewer

        # Channel choice               
        self.lab_ch = QLabel(self)
        self.lab_ch.setText('Channel')
        self.ch = QComboBox(self)
        self.ch.addItem('None')
        self.ch.addItem('0')
        self.ch.addItem('1')
        self.ch.addItem('2')
        self.ch.addItem('gray')
        self.ch.setFixedSize(185,20)
        self.lab_ch.move(10,35)
        self.ch.move(220,35)  
        self.btn_ch = QPushButton('OK',self)        
        self.btn_ch.setFixedSize(50,20)
        self.btn_ch.move(425,35) 
        self.btn_ch.clicked.connect(self.callback_channel)   

        #'↻'
        # smoothing          
        self.lab_smth = QLabel(self)
        self.lab_smth.setText('Smoothing')
        self.lab_smth.move(10,65)                       
        self.btn_smth = QPushButton('OK',self)
        self.btn_smth.setFixedSize(50,20)
        self.btn_smth.move(425,65) 
        self.btn_smth.clicked.connect(self.callback_smoothing)   
        self.slider_smth = QDoubleSpinBox()
        self.slider_smth.setMinimum(0)
        self.slider_smth.setMaximum(255)
        self.slider_smth.setFixedSize(185,20)
        self.slider_smth.move(220,65) 

        # difference_of_gaussians           
        self.lab_diff_gauss = QLabel(self)
        self.lab_diff_gauss.setText('Difference_of_Gaussians')
        self.lab_diff_gauss.move(10,95)
        self.lab_diff_gauss.setFixedSize(200,20)                       
        self.btn_diff_gauss = QPushButton('OK',self)
        self.btn_diff_gauss.setFixedSize(50,20)
        self.btn_diff_gauss.move(425,95) 
        self.btn_diff_gauss.clicked.connect(self.callback_diff_gauss)   
        self.slider_diff_gauss = QDoubleSpinBox()
        self.slider_diff_gauss.setMinimum(0.0)
        self.slider_diff_gauss.setMaximum(10.0)
        self.slider_diff_gauss.setFixedSize(60,20)
        self.slider_diff_gauss.move(220,95) 
        self.slider_diff_gauss2 = QDoubleSpinBox()
        self.slider_diff_gauss2.setMinimum(0.0)
        self.slider_diff_gauss2.setMaximum(10.0)
        self.slider_diff_gauss2.setFixedSize(60,20)
        self.slider_diff_gauss2.move(315,95)        

        self.btn_invert = QPushButton('   Invert  ',self)
        self.btn_invert.setFixedSize(185,20)
        self.btn_invert.move(220,125) 
        self.btn_invert.clicked.connect(self.callback_invert)

        self.btn_crp = QPushButton('   Crop  ',self)
        self.btn_crp.setFixedSize(185,20)
        self.btn_crp.move(220,155) 
        self.btn_crp.clicked.connect(self.callback_crop)           

        # difference_of_gaussians           
        self.lab_resize = QLabel(self)
        self.lab_resize.setText('resize an image')
        self.lab_resize.move(10,185)
        self.lab_resize.setFixedSize(200,20)                       
        self.btn_resize = QPushButton('OK',self)
        self.btn_resize.setFixedSize(50,20)
        self.btn_resize.move(425,185) 
        self.btn_resize.clicked.connect(self.callback_resize)   
        self.slider_resize = QSpinBox()
        self.slider_resize.setMinimum(0)
        self.slider_resize.setMaximum(65000)
        self.slider_resize.setFixedSize(60,20)
        self.slider_resize.move(220,185) 
        self.slider_resize2 = QSpinBox()
        self.slider_resize2.setMinimum(0)
        self.slider_resize2.setMaximum(65000)
        self.slider_resize2.setFixedSize(60,20)
        self.slider_resize2.move(315,185)        


        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self.lab_ch)
        self.layout().addWidget(self.ch)
        self.layout().addWidget(self.lab_diff_gauss)
        self.layout().addWidget(self.btn_ch)
        self.layout().addWidget(self.slider_diff_gauss)
        self.layout().addWidget(self.lab_smth)
        self.layout().addWidget(self.btn_smth)
        self.layout().addWidget(self.slider_smth)
        self.layout().addWidget(self.slider_diff_gauss2)        
        self.layout().addWidget(self.btn_crp)   
        self.layout().addWidget(self.btn_invert)   
        self.layout().addWidget(self.lab_resize)
        self.layout().addWidget(self.slider_resize)
        self.layout().addWidget(self.slider_resize2)
        self.layout().addWidget(self.btn_resize)

    def callback_channel(self):
        layers = list(self.viewer.layers)
        image = list(self.viewer.layers.selection)[0]
        for i, l in enumerate(layers):
            if l.name == 'channel_'+image.name:
                self.viewer.layers.pop(i)
        image = list(self.viewer.layers.selection)[0]
        self.channel = self.ch.currentText()
        if self.channel !='None':
            new_image = get_image(image.name,self.viewer,self.channel)
            self.viewer.add_image(new_image, name = 'channel_'+image.name)

        
    def callback_diff_gauss(self):
        self.min = self.slider_diff_gauss.value()
        self.max = self.slider_diff_gauss2.value()        
        layers = list(self.viewer.layers)
        for i, l in enumerate(layers):
            if l.name == 'difference_of_gaussians':
                self.viewer.layers.pop(i)
        image = list(self.viewer.layers.selection)[0]
        self.channel = self.ch.currentText()
        new_image = get_image(image.name,self.viewer,self.channel)
        filtered_image = difference_of_gaussians(new_image, self.min, self.max)    
        self.viewer.add_image(filtered_image, name = 'difference_of_gaussians')

    def callback_smoothing(self):
        self.segma = self.slider_smth.value()
        layers = list(self.viewer.layers)
        image = list(self.viewer.layers.selection)[0]
        self.channel = self.ch.currentText()
        grayscale = get_image(image.name,self.viewer,self.channel)
        for i, l in enumerate(layers):
            if l.name == 'gaussian':
                self.viewer.layers.pop(i)
        new_image = gaussian(grayscale,self.segma)  
        self.viewer.add_image(new_image, name = 'gaussian')        
    def callback_resize(self):
        self.high = self.slider_resize.value()
        self.width = self.slider_resize2.value()       
        self.channel = self.ch.currentText() 
        # layers = list(self.viewer.layers)
        # for i, l in enumerate(layers):
        #     if l.name == 'resized':0
        #         self.viewer.layers.pop(i)
        images = list(self.viewer.layers.selection)
        if  len(images) == 1:
            new_image = get_image(images[0].name,self.viewer,self.channel)
            resized = resize(new_image, (self.high, self.width),preserve_range=True)    
            self.viewer.add_image(np.uint64(resized), name = images[0].name + 'resized')
        else:
            for i in range(len(images)):
                image = images[i]
                new_image = get_image(image.name,self.viewer,self.channel)
                resized = resize(new_image, (self.high, self.width),preserve_range=True)    
                self.viewer.add_image(resized, name = image.name + 'resized')

    def callback_invert(self):
        layers = list(self.viewer.layers)
        image = list(self.viewer.layers.selection)[0]
        new_image = invert(image.data)  
        self.viewer.add_image(new_image, name = image.name +'inverted')
    def callback_crop(self):
        image = list(self.viewer.layers.selection)[0]                
        shape_layer = [{'types': layer_.shape_type, 'data': layer_.data}
                        for layer_ in self.viewer.layers if isinstance(layer_, Shapes)]
        self.channel = self.ch.currentText()
        image_name = image.name
        if self.channel == 'None':
            if image.data.ndim == 3 and len(shape_layer) != 0:
                image = image.data
            elif len(shape_layer)!=0:
                image = image.data.transpose(2,0,1)        
            if image.shape[0] < 36:
                print('image.shape[0] < 36')
                x, y = image.shape[1:3]
                for k in range(image.shape[0]):
                    for i in range(len(shape_layer)):
                        for j, type_ in enumerate(shape_layer[i]['types']):
                            data = shape_layer[i]['data'][j]
                            rect,x1,x2,y1,y2 = transform_to_rect(image[k], data, x, y)
                            self.viewer.add_image(np.uint(rect), name=image_name + 'croped_'+str(k)) 

        else:    
            grayscale = get_image(image.name,self.viewer,self.channel)
            x, y = grayscale.shape
            if len(shape_layer) == 0:
                self.viewer.add_image(grayscale, name='crop_')
            else:
                for i in range(len(shape_layer)):
                    for j, type_ in enumerate(shape_layer[i]['types']):
                        data = shape_layer[i]['data'][j]
                        rect,x1,x2,y1,y2 = transform_to_rect(grayscale, data, x, y)
                        self.viewer.add_image(np.uint(rect), name= image.name + 'crop_'+str(j)) 

class AnotherWindow(QWidget):
    """ This class widget  allows the user to set the parameters for Cellpose segmentation.

    This widget gives the user the ability to set the model parameters:
    """

    def __init__(self):
        super().__init__()
                    
        # model parameters    
        directory = Path(os.path.dirname(os.path.abspath(__file__)))
        model_path = str(directory.parent.absolute()) + '/pretrained models/cellpose/data.pkl'
        self.pretrained_pkl = model_path
        self.gpu = False
        self.model_type = 'None'
        self.diam_mean = 27
        #model evaluation parameters
        self.channels = [0, 0]
        self.flow_threshold = 0.4
        self.cellprob_threshold = 0.0
        self.min_size = 15

        self.Label_title = QLabel(self)
        self.Label_title.setText('model parameters')
        self.Label_title.setFixedSize(200,25)
        self.Label_title.move(10,5)

        self.Label_gpu = QLabel(self)
        self.Label_gpu.setText('GPU:')
        self.cbox_gpu = QComboBox(self)
        self.cbox_gpu.addItem('False')
        self.cbox_gpu.addItem('True')
        self.cbox_gpu.setFixedSize(200,25)
        self.Label_gpu.move(10,45)
        self.cbox_gpu.move(10,75)       

        self.Label_model_type = QLabel(self)
        self.Label_model_type.setText('model_type:')
        self.cbox_model_type = QComboBox(self)
        self.cbox_model_type.addItem('None')
        self.cbox_model_type.addItem('nuclei')
        self.cbox_model_type.addItem('cyto')
        self.cbox_model_type.setFixedSize(200,25)
        self.Label_model_type.move(10,105)
        self.cbox_model_type.move(10,135)       

        self.Label_diam_mean = QLabel(self)
        self.Label_diam_mean.setText('diam_mean:')
        self.qle_diam_mean = QLineEdit(self)
        self.qle_diam_mean.setPlaceholderText('Default: 27')
        self.qle_diam_mean.setFixedSize(200,25)
        self.Label_diam_mean.move(10,165)    
        self.qle_diam_mean.move(10,195)

        self.btn_pretrained = QPushButton('pretrained_pkl',self)
        self.btn_pretrained.setFixedSize(150,25)
        self.btn_pretrained.move(10,225)          
        self.btn_pretrained.clicked.connect(self.open_folder_in)        


        self.Label_title = QLabel(self)
        self.Label_title.setText('model evaluation parameters')
        self.Label_title.setFixedSize(300,25)
        self.Label_title.move(10,265)

        self.Label_channels = QLabel(self)
        self.Label_channels.setText('channels:')
        self.qle_channels = QLineEdit(self)
        self.qle_channels.setPlaceholderText('Default: [0, 0]')
        self.qle_channels.setFixedSize(200,25)
        self.Label_channels.move(10,305)
        self.qle_channels.move(10,335)       

        self.Label_flow_thresh = QLabel(self)
        self.Label_flow_thresh.setText('flow_threshold:')
        self.qle_flow_thresh = QLineEdit(self)
        self.qle_flow_thresh.setPlaceholderText('Default: 0.4')
        self.qle_flow_thresh.setFixedSize(200,25)
        self.Label_flow_thresh.move(10,365)
        self.qle_flow_thresh.move(10,395)       

        self.Label_cellprob_thresh = QLabel(self)
        self.Label_cellprob_thresh.setText('cellprob_threshold:')
        self.qle_cellprob_thresh = QLineEdit(self)
        self.qle_cellprob_thresh.setPlaceholderText('Default: 0.0')
        self.qle_cellprob_thresh.setFixedSize(200,25)
        self.Label_cellprob_thresh.move(10,425)
        self.qle_cellprob_thresh.move(10,455)
        
        self.Label_min_size = QLabel(self)
        self.Label_min_size.setText('min_size:')
        self.qle_min_size = QLineEdit(self)
        self.qle_min_size.setPlaceholderText('Default: 15')
        self.qle_min_size.setFixedSize(200,25)
        self.Label_min_size.move(10,485)
        self.qle_min_size.move(10,515)       

        self.btn_in_param = QPushButton('Input Parameters',self)
        self.btn_in_param.setFixedSize(150,25)
        self.btn_in_param.move(10,560)          
        self.btn_in_param.clicked.connect(self.button_click)
        

    def button_click(self):

        #model parameters
        v1 = self.cbox_gpu.currentText()
        if v1 == 'False':
            self.gpu = False
        else:
            self.gpu = True    
        v2 = self.cbox_model_type.currentText()
        if v2 == 'None':
            self.model_type = None
        else:
            self.model_type = v2    
        v3 = self.qle_diam_mean.text()
        if bool(v3):
            self.diam_mean = literal_eval(v3)

        #model evaluation parameters                
        v5 = self.qle_channels.text()
        if bool(v5):
            self.channels = literal_eval(v5) 
        v6 = self.qle_flow_thresh.text()
        if bool(v6):
            self.flow_threshold = literal_eval(v6) 
        v7 = self.qle_cellprob_thresh.text()
        if bool(v7):
            self.cellprob_threshold = literal_eval(v7) 
        v8 = self.qle_min_size.text()
        if bool(v8):
            self.min_size = literal_eval(v8) 

    def open_folder_in(self):
        dialog = QFileDialog()
        self.pretrained_pkl = QFileDialog.getOpenFileName()[0]

class AnotherWindow1(QWidget):
    """ This class widget  allows the user to set the parameters for Hough segmentation.

    This widget gives the user the ability to set the model parameters:
    """

    def __init__(self):
        super().__init__()
                    
        # model parameters    
        self.cell_mean = 60
        self.verbose = 'all'
        self.out_dir = os.getcwd()

        self.Label_title = QLabel(self)
        self.Label_title.setText('model parameters')
        self.Label_title.setFixedSize(200,25)
        self.Label_title.move(10,5)

        self.Label_vb = QLabel(self)
        self.Label_vb.setText('verbose:')
        self.cbox_vb = QComboBox(self)
        self.cbox_vb.addItem('False')
        self.cbox_vb.addItem('True')
        self.cbox_vb.addItem('all')        
        self.cbox_vb.setFixedSize(200,25)
        self.Label_vb.move(10,45)
        self.cbox_vb.move(10,75)

        self.Label_cell_mean = QLabel(self)
        self.Label_cell_mean.setText('cell_mean:')
        self.qle_cell_mean = QLineEdit(self)
        self.qle_cell_mean.setPlaceholderText('Default: 60')
        self.qle_cell_mean.setFixedSize(200,25)
        self.Label_cell_mean.move(10,105)    
        self.qle_cell_mean.move(10,135)

        self.btn_pretrained = QPushButton('output directory',self)
        self.btn_pretrained.setFixedSize(200,25)
        self.btn_pretrained.move(10,225)
        self.btn_pretrained.clicked.connect(self.open_folder_out)        

        self.btn_in_param = QPushButton('Input Parameters',self)
        self.btn_in_param.setFixedSize(200,25)
        self.btn_in_param.move(10,270)
        self.btn_in_param.clicked.connect(self.button_click)
        

    def button_click(self):
        #model parameters
        v1 = self.cbox_vb.currentText()        
        self.verbose = v1
        v2 = self.qle_cell_mean.text()
        if bool(v2):
            self.cell_mean = literal_eval(v2)
    def open_folder_out(self):
        dialog = QFileDialog()
        self.out_dir = dialog.getExistingDirectory(self, 'Select an awesome directory')

class segmentation (QMainWindow):
    """ This class widget  allows the user to do image segmentation.

    This widget gives the user the ability to:
    - Choose an image layer and a shape layer, and by clicking in update button the list of image/shape layers is updated. 
    - Choose a segmentation method:
        "simple_threshold" : to do segmentation based on classic algorithms.
        "advanced_threshold": to do segmentation based on advanced algorithms.
    - Choose a classic segmentation method:
        "otsu_threshold" : global image segmentation using Otsu thresholding. 
        "local_threshold" : local image segmentation using adaptive algorithm. 
        "manual_threshold" : global image segmentation based on user threshold choice. 
    - Choose a threshold : Used in case of manual_threshold type, s.t 0<threshold<255. 
    - Choose a block_size : The characteristic size surrounding each pixel, that defines a local neighborhoods
        on which a local or dynamic thresholding is calculated. 
    - Choose an advanced segmentation method:
        "StarDist": image segmentation using pretrained StarDist algorithm.
        "Cellpose": image segmentation using pretrained Cellpose algorithm.
    - Choose a StarDist_image_type :
        '2D_versatile_fluo' & '2D_paper_dsb2018': Versatile (fluorescent nuclei) and DSB 2018 
                                                (from StarDist 2D paper) that were both trained 
                                                on a subset of the DSB 2018 nuclei segmentation 
                                                challenge dataset.
        '2D_versatile_he': Versatile (H&E nuclei) that was trained on images from the MoNuSeg 2018 
                           training data and the TNBC dataset from Naylor et al. (2018).                                               
    - Set the parameters for Cellpose segmentation:

    """
    def __init__(self, napari_viewer, parent=None):
        """ QWidget.__init__ method.

        Parameters
        ----------
        napari_viewer : instance
            Access to napari viewer in order to add widgets.
        parent : class
            Parent class (QWidget)
        """
        super().__init__(parent)        
        self.viewer = napari_viewer
        self.window = AnotherWindow()
        self.window1 = AnotherWindow1()

        # Choose an image layer
        self.Label_choice_img = QLabel(self)
        self.Label_choice_img.setText('Image_layers')
        self.qle_choice_img = QComboBox(self)
        self.qle_choice_img.setFixedSize(200,25)
        self.Label_choice_img.move(10,10)            
        self.qle_choice_img.move(200,10)  

        # Choose a shape layer
        self.Label_choice_sh = QLabel(self)
        self.Label_choice_sh.setText('Shape_layers')
        self.qle_choice_sh = QComboBox(self)
        self.qle_choice_sh.addItem('')
        self.qle_choice_sh.setFixedSize(200,25)
        self.btn_choice = QPushButton('↻',self)
        self.btn_choice.setFixedSize(40,25)
        self.btn_choice.clicked.connect(self.callback_layers_update)
        self.Label_choice_sh.move(10,40)            
        self.qle_choice_sh.move(200,40)  
        self.btn_choice.move(430,25)

        # Choose the segmentation method
        self.Label_seg_method = QLabel(self)
        self.Label_seg_method.setText('segmentation method')
        self.Label_seg_method.setFixedSize(200,25)
        self.qle_seg_method = QComboBox(self)
        self.qle_seg_method.addItem('classic methods')
        self.qle_seg_method.addItem('advanced methods')
        self.qle_seg_method.setFixedSize(200,25)
        self.Label_seg_method.move(10,70)
        self.qle_seg_method.move(200,70) 
        self.qle_seg_method.currentTextChanged.connect(self.callback_method) 

        # Choose a classic segmentation method
        self.Label_classic = QLabel(self)
        self.Label_classic.setText('classic segmentation')
        self.qle_classic = QComboBox(self)
        self.qle_classic.addItem('manual thresholding')
        self.qle_classic.addItem('Otsu thresholding')
        self.qle_classic.addItem('local thresholding')
        self.qle_classic.addItem('Hough transform')
        self.qle_classic.setFixedSize(200,25)
        self.Label_classic.setFixedSize(200,25)
        self.Label_classic.move(10,100)
        self.qle_classic.move(200,100) 
        self.qle_classic.currentTextChanged.connect(self.callback_classic) 

        # Choose a threshold for manual thresholding
        self.Label_thresh = QLabel(self)
        self.Label_thresh.setText('Threshold')
        self.spbox_thresh = QDoubleSpinBox()
        self.spbox_thresh.setMinimum(0.0)
        self.spbox_thresh.setMaximum(255.0)
        self.spbox_thresh.setFixedSize(200,25)
        self.spbox_thresh.move(200,130) 
        self.Label_thresh.move(10,130) 

        # Choose a bloc size for local thresholding
        self.Label_bloc_size = QLabel(self)
        self.Label_bloc_size.setText('Bloc size')
        self.spbox_bloc_size = QSpinBox()
        self.spbox_bloc_size.setMinimum(0)
        self.spbox_bloc_size.setMaximum(1000)
        self.spbox_bloc_size.setFixedSize(200,25)
        self.spbox_bloc_size.move(200,130) 
        self.Label_bloc_size.move(10,130) 
        self.spbox_bloc_size.hide()
        self.Label_bloc_size.hide()

        # showing the Otsu threshold
        self.Label_otsu = QLabel(self)
        self.Label_otsu.setText('Otsu threshold')
        self.qle_otsu = QLineEdit()
        self.qle_otsu.setFixedSize(200,25)
        self.Label_otsu.setFixedSize(200,25)
        self.qle_otsu.move(200,130) 
        self.Label_otsu.move(10,130) 
        self.qle_otsu.hide()
        self.Label_otsu.hide()

        # Set the parameters for Hough transform
        self.btn_param_hough = QPushButton('set parameters',self)
        self.btn_param_hough.hide()
        self.btn_param_hough.setFixedSize(200,25)
        self.btn_param_hough.clicked.connect(self.callback_param_hough)
        self.btn_param_hough.move(200,130)
        self.btn_param_hough.hide()

        # Choose a advanced segmentation method
        self.Label_advanced = QLabel(self)
        self.Label_advanced.setText('advanced segmentation')
        self.qle_advanced = QComboBox(self)
        self.qle_advanced.addItem('StarDist')
        self.qle_advanced.addItem('cellpose')
        self.qle_advanced.setFixedSize(200,25)
        self.Label_advanced.setFixedSize(200,25)
        self.Label_advanced.move(10,100)
        self.qle_advanced.move(200,100)
        self.qle_advanced.currentTextChanged.connect(self.callback_advanced) 
        self.qle_advanced.hide()
        self.Label_advanced.hide()

        # Choose a advanced segmentation method
        self.Label_img_type = QLabel(self)
        self.Label_img_type.setText('StarDist image type')
        self.qle_img_type = QComboBox(self)
        self.qle_img_type.addItem('2D_versatile_fluo')
        self.qle_img_type.addItem('2D_versatile_he')
        self.qle_img_type.addItem('2D_paper_dsb2018')
        self.qle_img_type.setFixedSize(200,25)
        self.Label_img_type.setFixedSize(200,25)
        self.Label_img_type.move(10,130)
        self.qle_img_type.move(200,130)
        self.qle_img_type.hide()
        self.Label_img_type.hide()

        # Choose options for cellpose segmentation
        self.btn_param = QPushButton('set parameters',self)
        self.btn_param.hide()
        self.btn_param.setFixedSize(200,25)
        self.btn_param.clicked.connect(self.callback_param)
        self.btn_param.move(200,130)
        self.btn_param.hide()

        # Execution Button 
        self.btn_run = QPushButton('Run',self)
        self.btn_run.setFixedSize(250,25)
        self.btn_run.move(95,175)
        self.btn_run.clicked.connect(self.callback_run)
        
        self.setLayout(QHBoxLayout(self))
        self.layout().addWidget(self.Label_choice_img)
        self.layout().addWidget(self.qle_choice_img)
        self.layout().addWidget(self.Label_choice_sh)
        self.layout().addWidget(self.qle_choice_sh)
        self.layout().addWidget(self.btn_choice)
        self.layout().addWidget(self.Label_seg_method)
        self.layout().addWidget(self.qle_seg_method)
        self.layout().addWidget(self.Label_classic)
        self.layout().addWidget(self.qle_classic)
        self.layout().addWidget(self.spbox_thresh)
        self.layout().addWidget(self.Label_thresh)
        self.layout().addWidget(self.spbox_bloc_size)
        self.layout().addWidget(self.Label_bloc_size)
        self.layout().addWidget(self.Label_otsu)
        self.layout().addWidget(self.qle_otsu)
        self.layout().addWidget(self.Label_advanced)
        self.layout().addWidget(self.qle_advanced)
        self.layout().addWidget(self.Label_img_type)
        self.layout().addWidget(self.qle_img_type)
        self.layout().addWidget(self.btn_param)
        self.layout().addWidget(self.btn_run)

    def callback_layers_update(self):
        self.qle_choice_img.clear()
        self.qle_choice_sh.clear()
        self.qle_choice_sh.addItem('')
        layers = list(self.viewer.layers)
        for l in layers:
            if isinstance(l, napari.layers.Image):
                self.qle_choice_img.addItem(l.name)
            elif isinstance(l, napari.layers.Shapes):
                self.qle_choice_sh.addItem(l.name)    

    def callback_method(self, value):
        """ This function is responsible of showing and hidding the widgets.
        """ 

        if value =='classic methods':
            self.btn_param.hide()
            self.qle_img_type.hide()
            self.Label_img_type.hide()
            self.qle_advanced.hide()
            self.Label_advanced.hide()
            self.qle_otsu.hide()
            self.Label_otsu.hide()
            self.spbox_bloc_size.hide()
            self.Label_bloc_size.hide()
            self.Label_thresh.hide()
            self.spbox_thresh.hide()
            self.qle_classic.show()
            self.Label_classic.show()

        else:
            self.btn_param.hide()
            self.qle_img_type.hide()
            self.Label_img_type.hide()
            self.qle_advanced.show()
            self.Label_advanced.show()
            self.qle_otsu.hide()
            self.Label_otsu.hide()
            self.spbox_bloc_size.hide()
            self.Label_bloc_size.hide()
            self.Label_thresh.hide()
            self.spbox_thresh.hide()
            self.qle_classic.hide()
            self.Label_classic.hide()

    def callback_classic(self, value):
        """ This function is responsible of showing and hidding the widgets.
        """ 

        self.btn_param.hide()
        self.qle_img_type.hide()
        self.Label_img_type.hide()
        self.qle_advanced.hide()
        self.Label_advanced.hide()
        self.qle_otsu.hide()
        self.Label_otsu.hide()
        self.spbox_bloc_size.hide()
        self.Label_bloc_size.hide()
        self.Label_thresh.hide()
        self.spbox_thresh.hide()
        self.qle_classic.show()
        self.Label_classic.show()
        if value =='manual thresholding':
            self.Label_thresh.show()
            self.spbox_thresh.show()

        elif value == 'Otsu thresholding':
            self.qle_otsu.show()
            self.Label_otsu.show()
        elif value == 'local thresholding':
            self.spbox_bloc_size.show()
            self.Label_bloc_size.show()
        else:
            self.btn_param_hough.show()    

    def callback_advanced(self, value):
        """ This function is responsible of showing and hidding the widgets.
        """ 

        self.btn_param.hide()
        self.qle_img_type.hide()
        self.Label_img_type.hide()
        self.qle_advanced.show()
        self.Label_advanced.show()
        self.qle_otsu.hide()
        self.Label_otsu.hide()
        self.spbox_bloc_size.hide()
        self.Label_bloc_size.hide()
        self.Label_thresh.hide()
        self.spbox_thresh.hide()
        self.qle_classic.hide()
        self.Label_classic.hide()
        if value =='StarDist':
            self.qle_img_type.show()
            self.Label_img_type.show()
        elif value == 'cellpose':
            self.btn_param.show()

    def callback_param(self, checked):
        if self.window.isVisible():
            self.window.hide()
        else:
            self.window.show()

    def callback_param_hough(self, checked):
        if self.window1.isVisible():
            self.window1.hide()
        else:
            self.window1.show()

    def callback_run(self):
        """ This function is responsible of running the segmentation.
        """ 

        self.threshold = self.spbox_thresh.value()
        self.bloc_size = self.spbox_bloc_size.value()
        self.classic_method = self.qle_classic.currentText()
        self.seg_method = self.qle_seg_method.currentText()
        self.image_name = self.qle_choice_img.currentText()
        self.shape_name = self.qle_choice_sh.currentText()
        self.advanced_method = self.qle_advanced.currentText()
        self.StarDist_image_type = self.qle_img_type.currentText()

        annotator_shapes = AnnotatorShapes()
        if self.seg_method == "classic methods":
            if self.classic_method == "Otsu thresholding":
                Otsu_thresh = annotator_shapes.on_click(
                    self.viewer, 0, 0, self.image_name, self.shape_name)
                self.qle_otsu.setText(str(Otsu_thresh))        
            elif self.classic_method == "manual thresholding":
                Otsu_thresh = annotator_shapes.on_click(
                    self.viewer, self.threshold, 0, self.image_name, self.shape_name)
            elif self.classic_method == "local thresholding":
                Otsu_thresh = annotator_shapes.on_click(
                    self.viewer, 1, self.bloc_size, self.image_name, self.shape_name)
            else:
                self.param_hough = {"cell_mean":self.window1.cell_mean,"verbose":self.window1.verbose,"out_dir":self.window1.out_dir}
                hough_segmentation(self.viewer, self.image_name, self.shape_name,self.param_hough)
        else:
            if self.advanced_method == "StarDist":
                stardist_segmentation(self.viewer, self.image_name, self.shape_name, self.StarDist_image_type)  
            else: 
                self.param = {"pretrained_pkl":self.window.pretrained_pkl,"GPU":self.window.gpu,"model_type":self.window.model_type,
                                "diam_mean":self.window.diam_mean,"channels":self.window.channels,"flow_threshold":self.window.flow_threshold,
                                "cellprob_threshold":self.window.cellprob_threshold,"min_size":self.window.min_size}
                cellpose_segmentation(self.viewer, self.image_name, self.shape_name, self.param)

class AnotherWindow2(QWidget):
    """ This class widget  allows the user to set the parameters for Hough segmentation.

    This widget gives the user the ability to set the model parameters:
    """

    def __init__(self):
        super().__init__()
                    
        # model parameters    
        self.cell_mean = 60
        self.mask = True
        self.out_dir = os.getcwd()
        self.size = 71 

        self.Label_title = QLabel(self)
        self.Label_title.setText('model parameters')
        self.Label_title.setFixedSize(200,25)
        self.Label_title.move(10,5)

        self.Label_mask = QLabel(self)
        self.Label_mask.setText('mask:')
        self.cbox_mask = QComboBox(self)
        self.cbox_mask.addItem('False')
        self.cbox_mask.addItem('True')
        self.cbox_mask.setFixedSize(200,25)
        self.Label_mask.move(10,45)
        self.cbox_mask.move(10,75)

        self.Label_cell_mean = QLabel(self)
        self.Label_cell_mean.setText('cell_mean:')
        self.qle_cell_mean = QLineEdit(self)
        self.qle_cell_mean.setPlaceholderText('Default: 60')
        self.qle_cell_mean.setFixedSize(200,25)
        self.Label_cell_mean.move(10,105)    
        self.qle_cell_mean.move(10,135)

        self.Label_size = QLabel(self)
        self.Label_size.setText('box size:')
        self.qle_size = QLineEdit(self)
        self.qle_size.setPlaceholderText('Default: 71')
        self.qle_size.setFixedSize(200,25)
        self.Label_size.move(10,165)    
        self.qle_size.move(10,195)

        self.btn_pretrained = QPushButton('output directory',self)
        self.btn_pretrained.setFixedSize(200,25)
        self.btn_pretrained.move(10,235)
        self.btn_pretrained.clicked.connect(self.open_folder_out)        

        self.btn_in_param = QPushButton('Input Parameters',self)
        self.btn_in_param.setFixedSize(200,25)
        self.btn_in_param.move(10,280)
        self.btn_in_param.clicked.connect(self.button_click)
        

    def button_click(self):
        #model parameters
        v1 = self.cbox_mask.currentText()        
        if v1 == 'False':
            self.mask = False
        else:
            self.mask = True    
        v2 = self.qle_cell_mean.text()
        if bool(v2):
            self.cell_mean = literal_eval(v2)
        v3 = self.qle_size.text()
        if bool(v3):
            self.size = literal_eval(v3)

    def open_folder_out(self):
        dialog = QFileDialog()
        self.out_dir = dialog.getExistingDirectory(self, 'Select an awesome directory')

class Extract_imagette (QMainWindow):
    """ This class widget will extract rectangular imagette from the input image.

    This widget gives the user the ability to:
    - Set the parameters: 
    """
    def __init__(self, napari_viewer, parent=None):
        """ QWidget.__init__ method.

        Parameters
        ----------
        napari_viewer : instance
            Access to napari viewer in order to add widgets.
        parent : class
            Parent class (QWidget)
        """
        super().__init__(parent)        
        self.viewer = napari_viewer
        self.window2 = AnotherWindow2()

        # Choose an image layer
        self.Label_choice_img = QLabel(self)
        self.Label_choice_img.setText('Image layer')
        self.qle_choice_img = QComboBox(self)
        self.qle_choice_img.setFixedSize(200,25)
        self.btn_choice_img = QPushButton('↻',self)
        self.btn_choice_img.setFixedSize(40,25)
        self.btn_choice_img.clicked.connect(self.callback_image_choice)
        self.Label_choice_img.move(10,10)
        self.qle_choice_img.move(150,10)
        self.btn_choice_img.move(360,25)                  

        # Choose a Labels layer
        self.Label_choice_lab = QLabel(self)
        self.Label_choice_lab.setText('Labels layer')
        self.Label_choice_lab.setFixedSize(200,25)
        self.qle_choice_lab = QComboBox(self)
        self.qle_choice_lab.setFixedSize(200,25)
        self.Label_choice_lab.move(10,40)
        self.qle_choice_lab.move(150,40)

        # Set the parameters 
        self.Label_param = QLabel(self)
        self.Label_param.setText('set parameters')
        self.Label_param.setFixedSize(200,25)
        self.btn_param = QPushButton('set parameters',self)
        self.btn_param.setFixedSize(200,25)
        self.btn_param.clicked.connect(self.callback_set_param)
        self.Label_param.move(10,70)
        self.btn_param.move(150,70)

        # Execution Button 
        self.btn_run = QPushButton('Run',self)
        self.btn_run.setFixedSize(250,25)
        self.btn_run.move(95,110)          
        self.btn_run.clicked.connect(self.callback_run)


    def callback_image_choice(self):
        self.qle_choice_img.clear()
        layers = list(self.viewer.layers)
        for l in layers:
            if isinstance(l, napari.layers.Image):
                self.qle_choice_img.addItem(l.name)
        self.qle_choice_lab.clear()
        layers = list(self.viewer.layers)
        for l in layers:
            if isinstance(l, napari.layers.Labels):
                self.qle_choice_lab.addItem(l.name)

    def callback_set_param(self, checked):
        if self.window2.isVisible():
            self.window2.hide()
        else:
            self.window2.show()

    def callback_run(self):
        self.image_name = self.qle_choice_img.currentText()
        self.lab_name = self.qle_choice_lab.currentText()
        layers = list(self.viewer.layers)
        for l in layers:
            if isinstance(l, napari.layers.Image):
                if l.name == self.image_name:
                    self.image_data = l.data 
            elif isinstance(l, napari.layers.Labels):
                if l.name == self.lab_name:
                    self.lab_data = l.data
        print('start of extracting_cells')
        extract_cells(self.image_data, self.lab_data,raw = 0, coords_para = False, coords_distrac=False,cells_mean=self.window2.cell_mean , size1=self.window2.size, size2=self.window2.size, 
                     travel_output=self.window2.out_dir +"_output/", tophat=False, zoro=False,mask=self.window2.mask)
        print('end of extracting_cells')

######################################## Image Reconstruction (FPM) ########################################################

class AnotherWindow3(QWidget):
    """ This class widget  allows the user to set the parameters for Hough segmentation.

    This widget gives the user the ability to set the model parameters:
    """

    def __init__(self):
        super().__init__()
                    

        # model parameters    
        self.input_image_dir = os.getcwd()
        self.output_image_dir = os.getcwd()
        self.index_downSample = 2 # downsample: index_downSample=4
        self.crop = 1228
        self.arraysize = 5.97
        self.wlength = 0.620*1e-6
        self.NA = 0.75
        self.k0 = 2 * pi / self.wlength
        self.magnif = 20
        self.spsize = (3.45*1e-6)/self.magnif
        self.psize = self.spsize/self.index_downSample
        self.imSize = int(self.crop*self.index_downSample)
        self.imCenter = int(self.imSize / 2)
        self.NAstep = 0.05
        self.dz = 21

        self.Label_title = QLabel(self)
        self.Label_title.setText('model parameters')
        self.Label_title.setFixedSize(200,25)
        self.Label_title.move(10,5)

        self.Label_in_dwns = QLabel(self)
        self.Label_in_dwns.setText('index_downSample:')
        self.qle_in_dwns = QLineEdit(self)
        self.qle_in_dwns.setPlaceholderText('Default: 2')
        self.qle_in_dwns.setFixedSize(200,25)
        self.Label_in_dwns.move(10,45)
        self.qle_in_dwns.move(10,75)

        self.Label_im_size = QLabel(self)
        self.Label_im_size.setText('image size:')
        self.qle_im_size = QLineEdit(self)
        self.qle_im_size.setPlaceholderText('Default: 1228')
        self.qle_im_size.setFixedSize(200,25)
        self.Label_im_size.move(10,105)    
        self.qle_im_size.move(10,135)

        self.Label_size = QLabel(self)
        self.Label_size.setText('arraysize:')
        self.qle_size = QLineEdit(self)
        self.qle_size.setPlaceholderText('Default: 5.97')
        self.qle_size.setFixedSize(200,25)
        self.Label_size.move(10,165)    
        self.qle_size.move(10,195)

        self.btn_in_img = QPushButton('input image directory',self)
        self.btn_in_img.setFixedSize(200,25)
        self.btn_in_img.move(10,235)
        self.btn_in_img.clicked.connect(self.open_folder_in)        

        self.btn_out_img = QPushButton('output image directory',self)
        self.btn_out_img.setFixedSize(200,25)
        self.btn_out_img.move(10,270)
        self.btn_out_img.clicked.connect(self.open_folder_out)        

        self.btn_pretrained = QPushButton('save weights directory',self)
        self.btn_pretrained.setFixedSize(200,25)
        self.btn_pretrained.move(10,305)
        self.btn_pretrained.clicked.connect(self.save_weights_dir)        

        self.btn_in_param = QPushButton('Input Parameters',self)
        self.btn_in_param.setFixedSize(200,25)
        self.btn_in_param.move(10,340)
        self.btn_in_param.clicked.connect(self.button_click)
        

    def button_click(self):
        #model parameters
        v1 = self.qle_in_dwns.text()        
        if bool(v1):
            self.index_downSample = literal_eval(v1)
        v2 = self.qle_im_size.text()
        if bool(v2):
            self.crop = literal_eval(v2)
        v3 = self.qle_size.text()
        if bool(v3):
            self.arraysize = literal_eval(v3)

    def open_folder_in(self):
        dialog = QFileDialog()
        self.input_image_dir = dialog.getExistingDirectory(self, 'Select an awesome directory')
    def open_folder_out(self):
        dialog = QFileDialog()
        self.output_image_dir = dialog.getExistingDirectory(self, 'Select an awesome directory')
    def save_weights_dir(self):
        dialog = QFileDialog()
        self.save_weights_dir = dialog.getExistingDirectory(self, 'Select an awesome directory')



class Image_reconstruction (QMainWindow):
    """ This class widget will extract rectangular imagette from the input image.

    This widget gives the user the ability to:
    - Set the parameters: 
    """
    def __init__(self, napari_viewer, parent=None):
        """ QWidget.__init__ method.

        Parameters
        ----------
        napari_viewer : instance
            Access to napari viewer in order to add widgets.
        parent : class
            Parent class (QWidget)
        """
        super().__init__(parent)        
        self.viewer = napari_viewer
        self.window3 = AnotherWindow3()


        # Choose either testing or training the model
        self.qc_testing = QCheckBox("Testing",self)
        self.qc_testing.setChecked(True)
        self.qc_testing.stateChanged.connect(lambda:self.btnstate(self.qc_testing))
        self.qc_testing.setFixedSize(150,25)
        self.qc_testing.move(40,10)
        self.qc_training = QCheckBox("Training",self)
        self.qc_training.setChecked(False)
        self.qc_training.stateChanged.connect(lambda:self.btnstate(self.qc_training))
        self.qc_training.setFixedSize(150,25)
        self.qc_training.move(210,10)

        # Choose an image layer
        self.Label_choice_img = QLabel(self)
        self.Label_choice_img.setText('Image layer')
        self.qle_choice_img = QComboBox(self)
        self.qle_choice_img.setFixedSize(200,25)
        self.btn_choice_img = QPushButton('↻',self)
        self.btn_choice_img.setFixedSize(40,25)
        self.btn_choice_img.clicked.connect(self.callback_image_choice)
        self.Label_choice_img.move(10,40)
        self.qle_choice_img.move(220,40)
        self.btn_choice_img.move(430,40)                  

        # Load a pretrained model 
        self.Label_path = QLabel(self)
        self.Label_path.setText('path to pretrained model')
        self.Label_path.setFixedSize(200,25)
        self.btn_path = QPushButton('path',self)
        self.btn_path.setFixedSize(200,25)
        self.btn_path.clicked.connect(self.callback_set_path)
        self.Label_path.move(10,75)
        self.btn_path.move(220,75)

        self.Label_out_img = QLabel(self)
        self.Label_out_img.setText('output image directory')
        self.Label_out_img.setFixedSize(200,25)
        self.btn_out_img = QPushButton('Path',self)
        self.btn_out_img.setFixedSize(200,25)
        self.btn_out_img.move(220,105)
        self.Label_out_img.move(10,105)
        self.btn_out_img.clicked.connect(self.open_folder_out)        


        # Set the parameters 
        self.Label_param = QLabel(self)
        self.Label_param.setText('set parameters')
        self.Label_param.setFixedSize(200,25)
        self.btn_param = QPushButton('set parameters',self)
        self.btn_param.setFixedSize(200,25)
        self.btn_param.clicked.connect(self.callback_set_param)
        self.Label_param.move(10,70)
        self.btn_param.move(150,70)
        self.Label_param.hide()
        self.btn_param.hide()

        # Execution Button 
        self.btn_run = QPushButton('Run',self)
        self.btn_run.setFixedSize(250,25)
        self.btn_run.move(95,140)          
        self.btn_run.clicked.connect(self.callback_run)

    def open_folder_out(self):
        dialog = QFileDialog()
        self.out_dir = dialog.getExistingDirectory(self, 'Select an awesome directory')

    def btnstate(self,b):
        if b.text() == "Training":
            if b.isChecked() == True:
                self.Label_param.show()
                self.btn_param.show()
                self.Label_path.hide()
                self.btn_path.hide()
                self.Label_choice_img.hide()
                self.qle_choice_img.hide()
                self.btn_choice_img.hide()
                self.qc_testing.setChecked(False)
                self.Label_out_img.hide()
                self.btn_out_img.hide()        
            else:
                self.qc_training.setChecked(False)
                self.Label_param.hide()
                self.btn_param.hide()
        if b.text() == "Testing":
            if b.isChecked() == True:
                self.Label_param.hide()
                self.btn_param.hide()
                self.Label_choice_img.show()
                self.qle_choice_img.show()
                self.btn_choice_img.show()                  
                self.Label_path.show()
                self.btn_path.show()
                self.qc_training.setChecked(False)
                self.Label_out_img.show()
                self.btn_out_img.show()        
            else:
                self.qc_testing.setChecked(False)
                self.Label_choice_img.hide()
                self.qle_choice_img.hide()
                self.btn_choice_img.hide()                  
                self.Label_path.hide()
                self.btn_path.hide()

    def callback_image_choice(self):
        self.qle_choice_img.clear()
        layers = list(self.viewer.layers)
        for l in layers:
            if isinstance(l, napari.layers.Image):
                self.qle_choice_img.addItem(l.name)

    def callback_set_param(self, checked):
        if self.window3.isVisible():
            self.window3.hide()
        else:
            self.window3.show()
    def callback_set_path(self):
        dialog = QFileDialog()
        self.pretrained_h5 = QFileDialog.getOpenFileName()[0]

    def callback_run(self):
        self.image_name = self.qle_choice_img.currentText()
        layers = list(self.viewer.layers)
        for l in layers:
            if isinstance(l, napari.layers.Image):
                if l.name == self.image_name:
                    self.image_data = l.data 
        if self.qc_testing.isChecked() == True:
            model = create_models(self.window3.imSize, self.window3.crop,self.window3.index_downSample)
            model.load_weights(self.pretrained_h5)
            # model.predict(self.image_data)
            show_result(self.viewer,self.out_dir, model, self.window3.imSize, show=1, noShow=10,size=55)
            print('testing')
        elif self.qc_training.isChecked() == True:
            print('training')
            reconstruction_function(self.viewer, self.window3.input_image_dir ,self.window3.output_image_dir,self.window3.save_weights_dir,self.window3.arraysize,
                                self.window3.imSize,self.window3.crop,self.window3.index_downSample,self.window3.wlength,
                                self.window3.NA,self.window3.k0,self.window3.imCenter,self.window3.psize)

######################################## Image Classififcation ########################################################

class AnotherWindow4(QWidget):
    """ This class widget  allows the user to set the parameters for Hough segmentation.

    This widget gives the user the ability to set the model parameters:
    """

    def __init__(self):
        super().__init__()
                    

        # model parameters    
        self.input_image_dir = os.getcwd()
        self.output_image_dir = os.getcwd()
        self.lr = 0.00001 
        self.nb_epochs = 100
        self.batch_size = 20
        self.split_train_val = 0.8

        self.Label_title = QLabel(self)
        self.Label_title.setText('model parameters')
        self.Label_title.setFixedSize(200,25)
        self.Label_title.move(10,5)

        self.Label_in_dwns = QLabel(self)
        self.Label_in_dwns.setText('learning rate:')
        self.qle_in_dwns = QLineEdit(self)
        self.qle_in_dwns.setPlaceholderText('Default: 0.00001')
        self.qle_in_dwns.setFixedSize(200,25)
        self.Label_in_dwns.move(10,45)
        self.qle_in_dwns.move(10,75)

        self.Label_im_size = QLabel(self)
        self.Label_im_size.setText('number of epochs:')
        self.qle_im_size = QLineEdit(self)
        self.qle_im_size.setPlaceholderText('Default: 100')
        self.qle_im_size.setFixedSize(200,25)
        self.Label_im_size.move(10,105)    
        self.qle_im_size.move(10,135)

        self.Label_size = QLabel(self)
        self.Label_size.setText('batch size:')
        self.qle_size = QLineEdit(self)
        self.qle_size.setPlaceholderText('Default: 20')
        self.qle_size.setFixedSize(200,25)
        self.Label_size.move(10,165)    
        self.qle_size.move(10,195)

        self.Label_split = QLabel(self)
        self.Label_split.setText('split train/(val+test):')
        self.qle_split = QLineEdit(self)
        self.qle_split.setPlaceholderText('Default: 0.8')
        self.qle_split.setFixedSize(200,25)
        self.Label_split.move(10,235)    
        self.qle_split.move(10,265)

        self.btn_in_img = QPushButton('input image directory',self)
        self.btn_in_img.setFixedSize(200,25)
        self.btn_in_img.move(10,300)
        self.btn_in_img.clicked.connect(self.open_folder_in)        

        self.btn_out_img = QPushButton('output directory',self)
        self.btn_out_img.setFixedSize(200,25)
        self.btn_out_img.move(10,335)
        self.btn_out_img.clicked.connect(self.open_folder_out)        


        self.btn_in_param = QPushButton('Input Parameters',self)
        self.btn_in_param.setFixedSize(200,25)
        self.btn_in_param.move(10,375)
        self.btn_in_param.clicked.connect(self.button_click)
        

    def button_click(self):
        #model parameters
        self.lr = 0.00001 
        self.nb_epochs = 100
        self.batch_size = 20
        self.split_train_val = 0.8

        v1 = self.qle_in_dwns.text()        
        if bool(v1):
            self.lr = literal_eval(v1)
        v2 = self.qle_im_size.text()
        if bool(v2):
            self.nb_epochs = literal_eval(v2)
        v3 = self.qle_size.text()
        if bool(v3):
            self.batch_size = literal_eval(v3)
        v4 = self.qle_split.text()
        if bool(v4):
                self.split_train_val = literal_eval(v4)

    def open_folder_in(self):
        dialog = QFileDialog()
        self.input_image_dir = dialog.getExistingDirectory(self, 'Select an awesome directory')
    def open_folder_out(self):
        dialog = QFileDialog()
        self.output_image_dir = dialog.getExistingDirectory(self, 'Select an awesome directory')

class Image_classification (QMainWindow):
    """ This class widget will extract rectangular imagette from the input image.

    This widget gives the user the ability to:
    - Set the parameters: 
    """
    def __init__(self, napari_viewer, parent=None):
        """ QWidget.__init__ method.

        Parameters
        ----------
        napari_viewer : instance
            Access to napari viewer in order to add widgets.
        parent : class
            Parent class (QWidget)
        """
        super().__init__(parent)        
        self.viewer = napari_viewer
        self.window4 = AnotherWindow4()
        self.input_image_dir = '/home/slimane/Desktop/all/design project/slimane_lio/out_classification2021-9-4/ensemble/resnet_152 3 epochs combo dataset imagenet G RAW_0 partition.mat'


        self.qc_testing = QCheckBox("Testing",self)
        self.qc_testing.setChecked(True)
        self.qc_testing.stateChanged.connect(lambda:self.btnstate(self.qc_testing))
        self.qc_testing.setFixedSize(150,25)
        self.qc_testing.move(40,10)

        self.qc_imgext = QCheckBox("external_image",self)
        self.qc_imgext.setChecked(False)
        self.qc_imgext.stateChanged.connect(lambda:self.btnstate1(self.qc_imgext))
        self.qc_imgext.setFixedSize(150,25)
        self.qc_imgext.move(40,40)
        self.qc_imgint = QCheckBox("internal_image",self)
        self.qc_imgint.setChecked(True)
        self.qc_imgint.stateChanged.connect(lambda:self.btnstate1(self.qc_imgint))
        self.qc_imgint.setFixedSize(150,25)
        self.qc_imgint.move(250,40)

        # Choose a testing images folder
        self.Label_in_img = QLabel(self)
        self.Label_in_img.setText('testing images directory')
        self.Label_in_img.setFixedSize(200,25)
        self.btn_in_img = QPushButton('path to .mat',self)
        self.btn_in_img.setFixedSize(200,25)
        self.btn_in_img.move(250,75)
        self.Label_in_img.move(10,75)
        self.btn_in_img.clicked.connect(self.open_folder_in)        
        self.Label_in_img.hide()
        self.btn_in_img.hide()

        # Show false positives
        self.Label_fp = QLabel(self)
        self.Label_fp.setText('False positives')
        self.qle_fp = QComboBox(self)
        self.qle_fp.addItem(' ')
        self.qle_fp.setFixedSize(200,25)
        self.Label_fp.move(10,115)
        self.qle_fp.move(150,115) 
        self.btn_fp = QPushButton('show',self)
        self.btn_fp.setFixedSize(70,25)
        self.btn_fp.move(425,115)
        self.btn_fp.clicked.connect(self.show_fp)        
        self.btn_choice = QPushButton('↻',self)
        self.btn_choice.setFixedSize(40,25)
        self.btn_choice.clicked.connect(self.callback_choice)
        self.btn_choice.move(370,130)
        self.qle_fp.hide()
        self.btn_fp.hide()
        self.Label_fp.hide()
        self.btn_choice.hide()

        # Show false positives
        self.Label_fneg = QLabel(self)
        self.Label_fneg.setText('False negatives')
        self.qle_fneg = QComboBox(self)
        self.qle_fneg.addItem(' ')
        self.qle_fneg.setFixedSize(200,25)
        self.Label_fneg.move(10,150)
        self.qle_fneg.move(150,150) 
        self.btn_fneg = QPushButton('show',self)
        self.btn_fneg.setFixedSize(70,25)
        self.btn_fneg.move(425,150)
        self.btn_fneg.clicked.connect(self.show_fneg)        
        self.qle_fneg.hide()
        self.btn_fneg.hide()
        self.Label_fneg.hide()

        # Choose an image layer        
        self.Label_choice_img = QLabel(self)
        self.Label_choice_img.setText('Image layer')
        self.qle_choice_img = QComboBox(self)
        self.qle_choice_img.setFixedSize(200,25)
        self.btn_choice_img = QPushButton('↻',self)
        self.btn_choice_img.setFixedSize(40,25)
        self.btn_choice_img.clicked.connect(self.callback_image_choice)
        self.Label_choice_img.move(10,75)
        self.qle_choice_img.move(250,75)
        self.btn_choice_img.move(460,75)                  

        self.Label_pred = QLabel(self)
        self.Label_pred.setText('The class of prdicted cell:')
        self.Label_pred.setFixedSize(200,25)
        self.qle_pred = QLineEdit(self)
        self.qle_pred.setFixedSize(200,25)
        self.Label_pred.move(10,115)    
        self.qle_pred.move(250,115)    

        self.qc_training = QCheckBox("Training",self)
        self.qc_training.setChecked(False)
        self.qc_training.stateChanged.connect(lambda:self.btnstate(self.qc_training))
        self.qc_training.setFixedSize(150,25)
        self.qc_training.move(250,10)
        self.btn_training = QPushButton('parameters setting',self)
        self.btn_training.setFixedSize(200,25)
        self.btn_training.move(250,50)
        self.btn_training.clicked.connect(self.callback_set_param)
        self.btn_training.hide()

        # Load a pretrained model 
        self.btn_run = QPushButton('run classification',self)
        self.btn_run.setFixedSize(200,25)
        self.btn_run.move(150,185)
        self.btn_run.clicked.connect(self.callback_run)

        self.image_neg = []
        self.image_pos = []
    def show_fneg(self):
        image = [im for im in (self.image_neg) if im.split('/')[-1] == self.qle_fneg.currentText()] 
        print('image',image)
        if image[0]!=' ':
            image = imread(image[0])
            self.viewer.add_image(image,name=self.qle_fneg.currentText())
    def show_fp(self):
        image = [im for im in (self.image_pos) if im.split('/')[-1] == self.qle_fp.currentText()] 
        print('image',image)
        if image[0]!=' ':
            image = imread(image[0])
            self.viewer.add_image(image,name=self.qle_fp.currentText())

    def callback_choice(self):
        print('callback_fp',len(self.image_pos))
        self.qle_fp.clear()
        for l in self.image_pos:
            if path.exists(l):
                self.qle_fp.addItem(l.split('/')[-1])
                print('fp_done')
        print('callback_fneg',len(self.image_neg))
        self.qle_fneg.clear()
        for l in self.image_neg:
            if path.exists(l):
                self.qle_fneg.addItem(l.split('/')[-1])
        print('fneg_done')

    def callback_image_choice(self):
        self.qle_choice_img.clear()
        layers = list(self.viewer.layers)
        for l in layers:
            if isinstance(l, napari.layers.Image):
                self.qle_choice_img.addItem(l.name)
    def open_folder_in(self):
        dialog = QFileDialog()
        self.input_image_dir = QFileDialog.getOpenFileName()[0]
    def prediction(self, model,image):
        print('prediction')
        if image.ndim < 3:
            print('dim<3')
            im = np.stack((image,)*3, axis=-1)
            x,y,c = im.shape
            image = im.reshape((1,x,y,c))
            print(image.shape)
            pred = model.predict(image)
            pred = np.argmax(pred)
            classe = 'Infected' if pred == 1 else 'Healthy'
            print('pred',pred)
        elif image.ndim == 3:
            if image.shape[0] == 3 or image.shape[2] == 3:
                print('dim == 3')
                x,y,c = image.shape
                image = image.reshape((1,x,y,c))
                pred = model.predict(image)
                pred = np.argmax(pred)
                classe = 'Infected' if pred == 1 else 'Healthy'
                print('pred',pred,classe)
            else:
                classe = 'shape Error' 
                print('Only images of shape[3,:,:] or [: , :]')
        else:
            classe = 'shape Error' 
            print('Only images of shape[3,:,:] or [: , :]')                            
        self.qle_pred.setText(classe)

    def confusion_matrix(self,model,testing_generator,partition):
        scores = model.evaluate_generator(generator=testing_generator)
        print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

        Y_pred = model.predict_generator(generator=testing_generator)
        Y_pred = np.argmax(Y_pred, axis=1)
        Y_test = get_labels(partition['test'])[:len(Y_pred)]
        Y_test = Y_test.reshape([len(Y_test),])
        Y_test = np.array(Y_test, dtype='uint64')
        matrix = confusion_matrix(Y_test, Y_pred)
        plt.figure()
        plt.imshow(matrix)
        plt.title("matrice confusion")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(i,j,str(matrix[j,i]))
        plt.colorbar()
        plt.axis('off')
        plt.xlabel("prediction")
        plt.ylabel("reality")
        plt.show()

        #%%visualisation error

        f_neg = np.where(Y_test > Y_pred)
        f_pos = np.where(Y_test < Y_pred)
        part=np.array(partition['test'])
        image_neg_name = part[f_neg]
        image_pos_name = part[f_pos]
        for i,path in enumerate(image_neg_name):
            self.image_neg.append(path.split('.')[0] + '.' + path.split('.')[1][:4])
        for i,path in enumerate(image_pos_name):    
            self.image_pos.append(path.split('.')[0] + '.' + path.split('.')[1][:4])
        dicto = {'image_pos':self.image_pos,'image_neg':self.image_neg}
        from scipy.io import savemat
        savemat('/home/slimane/Desktop/all/design project/slimane_lio/classification/fp_fn.mat',dicto)
        print('self.image_pos',self.image_pos)
        print('self.image_neg',self.image_neg)
    def callback_run(self):
        if self.qc_testing.isChecked() == True:
            self.image_name = self.qle_choice_img.currentText()
            layers = list(self.viewer.layers)
            for l in layers:
                if isinstance(l, napari.layers.Image):
                    if l.name == self.image_name:
                        self.image_data = l.data 
            print('testing')
            input_shape=(84,84,3)
            archi = 'resnet_152' 
            num_cla = 2
            batch_size=20
            LEDS=[False,BF3,DF8,BF11,ALL,'multi_led']
            LED = LEDS[0]
            partition = mio.loadmat(self.input_image_dir)
            path = '/home/slimane/Desktop/all/design project/slimane_lio/out_classification/out_classification2021-9-5/checkpointer/resnet_152 epoch=002-val_acc=0.99.hdf5'
            model = create_model(archi, input_shape, num_cla, pre_train='imagenet')
            opt= Adam(lr, momentum)
            model._name="modelo"
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.load_weights(path)
            if self.qc_imgint.isChecked() == True:
                self.prediction(model,self.image_data)
            elif self.qc_imgext.isChecked() == True:
                testing_generator = DataGeneratorPhase(partition['test'], batch_size=batch_size, dim=input_shape[:2], 
                                    n_channels=input_shape[2], n_classes=num_cla, shuffle=False, LEDS=LED) 
                self.confusion_matrix(model,testing_generator,partition)    
        elif self.qc_training.isChecked() == True:
            print('training')
            classify()
    def btnstate(self,b):
        if b.text() == "Training":
            if b.isChecked() == True:
                self.btn_training.show()
                self.qle_fp.hide()
                self.btn_fp.hide()
                self.Label_fp.hide()
                self.btn_choice.hide()
                self.qle_fneg.hide()
                self.btn_fneg.hide()
                self.Label_fneg.hide()
                self.qc_imgext.hide()
                self.qc_imgint.hide()
                self.Label_choice_img.hide()
                self.qle_choice_img.hide()
                self.btn_choice_img.hide()
                self.Label_pred.hide()    
                self.qle_pred.hide()    
                self.Label_in_img.hide()
                self.btn_in_img.hide()
                self.qc_testing.setChecked(False)
            else:
                self.qc_training.setChecked(False)
                self.btn_training.hide()
        if b.text() == "Testing":
            if b.isChecked() == True:
                self.btn_training.hide()
                self.qle_fp.hide()
                self.btn_fp.hide()
                self.Label_fp.hide()
                self.btn_choice.hide()
                self.qle_fneg.hide()
                self.btn_fneg.hide()
                self.Label_fneg.hide()
                self.qc_imgext.show()
                self.qc_imgint.show()
                self.Label_choice_img.show()
                self.qle_choice_img.show()
                self.btn_choice_img.show()
                self.Label_pred.show()    
                self.qle_pred.show()    
                self.qc_imgint.setChecked(True)
                self.qc_training.setChecked(False)
            else:
                self.qc_testing.setChecked(False)
                self.qc_imgext.hide()
                self.qc_imgint.hide()
                self.Label_choice_img.hide()
                self.qle_choice_img.hide()
                self.btn_choice_img.hide()
                self.Label_pred.hide()    
                self.qle_pred.hide()    
                self.qle_fp.hide()
                self.btn_choice.hide()
                self.btn_fp.hide()
                self.Label_fp.hide()
                self.qle_fneg.hide()
                self.btn_fneg.hide()
                self.Label_fneg.hide()

    def btnstate1(self,b):
        if b.text() == "internal_image":
            if b.isChecked() == True:
                self.Label_choice_img.show()
                self.qle_choice_img.show()
                self.btn_choice_img.show()
                self.Label_pred.show()    
                self.qle_pred.show()    
                self.Label_in_img.hide()
                self.btn_in_img.hide()
                self.qle_fp.hide()
                self.btn_choice.hide()
                self.btn_fp.hide()
                self.Label_fp.hide()
                self.qle_fneg.hide()
                self.btn_fneg.hide()
                self.Label_fneg.hide()
                self.qc_imgext.setChecked(False)
            else:
                self.Label_pred.hide()    
                self.qle_pred.hide()    
                self.Label_choice_img.hide()
                self.qle_choice_img.hide()
                self.btn_choice_img.hide()
                self.qc_imgint.setChecked(False)
        if b.text() == "external_image":
            if b.isChecked() == True:
                self.Label_choice_img.hide()
                self.qle_choice_img.hide()
                self.btn_choice_img.hide()
                self.Label_pred.hide()    
                self.qle_pred.hide()    
                self.Label_in_img.show()
                self.btn_in_img.show()
                self.qle_fp.show()
                self.btn_fp.show()
                self.btn_choice.show()
                self.Label_fp.show()
                self.qle_fneg.show()
                self.btn_fneg.show()
                self.Label_fneg.show()
                self.qc_imgint.setChecked(False)
            else:
                self.Label_in_img.hide()
                self.btn_in_img.hide()
                self.qle_fp.hide()
                self.btn_fp.hide()
                self.Label_fp.hide()
                self.btn_choice.hide()
                self.qle_fneg.hide()
                self.btn_fneg.hide()
                self.Label_fneg.hide()
                self.qc_imgext.setChecked(False)

    def callback_set_param(self, checked):
        if self.window4.isVisible():
            self.window4.hide()
        else:
            self.window4.show()



################################################# Contour Assist ##################################################################
class contour_assist (QMainWindow):
    """ This class widget will contour assist the user during image segmentation.

    This widget gives the user the ability to:
    - Choose either options: 
        - contour suggestion: Based on the selected seeds, an image segmentation is performed using RegionGrowing algorithm.
        - add modification:  This function does an assemblage of labels in one label image.
    - Choose a threshold that defines the gray difference tolerated between a seed and the neighboring pixels.
    - Choose an image layer. 
    - Choose a raw label layer that includes the origin label layer.
    - Choose a modified label layer that includes the label layer to assemble with the origin one.
    """
    def __init__(self, napari_viewer, parent=None):
        """ QWidget.__init__ method.

        Parameters
        ----------
        napari_viewer : instance
            Access to napari viewer in order to add widgets.
        parent : class
            Parent class (QWidget)
        """
        super().__init__(parent)        
        self.viewer = napari_viewer

        # Choose the option
        self.Label_option = QLabel(self)
        self.Label_option.setText('Option')
        self.qle_option = QComboBox(self)
        self.qle_option.addItem('contour suggestion')
        self.qle_option.addItem('add modification')
        self.qle_option.setFixedSize(200,25)
        self.Label_option.move(10,10)
        self.qle_option.move(150,10) 
        self.qle_option.currentTextChanged.connect(self.callback_option) 

        # Choose an image layer
        self.Label_choice_img = QLabel(self)
        self.Label_choice_img.setText('Image_layers')
        self.qle_choice_img = QComboBox(self)
        self.qle_choice_img.setFixedSize(200,25)
        self.btn_choice_img = QPushButton('↻',self)
        self.btn_choice_img.setFixedSize(40,25)
        self.btn_choice_img.clicked.connect(self.callback_image_choice)

        # choose a threshold 
        self.Label_thresh = QLabel(self)
        self.Label_thresh.setText('Threshold')
        self.spbox_thresh = QDoubleSpinBox()
        self.spbox_thresh.setMinimum(0.0)
        self.spbox_thresh.setMaximum(10000.0)
        self.spbox_thresh.setFixedSize(200,25)

        self.Label_choice_img.move(10,40)            
        self.qle_choice_img.move(150,40)  
        self.btn_choice_img.move(360,40)
        self.Label_thresh.move(10,70)  
        self.spbox_thresh.move(150,70) 

        # Choose a raw Labels layer
        self.Label_choice_labr = QLabel(self)
        self.Label_choice_labr.setText('raw label')
        self.Label_choice_labr.hide()
        self.qle_choice_lab_raw = QComboBox(self)
        self.qle_choice_lab_raw.setFixedSize(200,25)
        self.qle_choice_lab_raw.hide()

        # Choose a modified Labels layer
        self.Label_choice_labm = QLabel(self)
        self.Label_choice_labm.setText('modified label')
        self.Label_choice_labm.setFixedSize(200,25)
        self.Label_choice_labm.hide()
        self.qle_choice_lab_mod = QComboBox(self)
        self.qle_choice_lab_mod.setFixedSize(200,25)
        self.qle_choice_lab_mod.hide()
        self.btn_choice_labm = QPushButton('↻',self)
        self.btn_choice_labm.hide()
        self.btn_choice_labm.setFixedSize(40,25)
        self.btn_choice_labm.clicked.connect(self.callback_label_choice)

        # Execution Button 
        self.btn_run = QPushButton('Run',self)
        self.btn_run.setFixedSize(250,25)
        self.btn_run.move(95,130)          
        self.btn_run.clicked.connect(self.callback_run)

        self.setLayout(QHBoxLayout(self))
        self.layout().addWidget(self.Label_option)
        self.layout().addWidget(self.qle_option)
        self.layout().addWidget(self.Label_choice_img)
        self.layout().addWidget(self.qle_choice_img)
        self.layout().addWidget(self.btn_choice_img)
        self.layout().addWidget(self.spbox_thresh)
        self.layout().addWidget(self.Label_thresh)
        self.layout().addWidget(self.Label_choice_labr)
        self.layout().addWidget(self.qle_choice_lab_raw)
        self.layout().addWidget(self.Label_choice_labm)
        self.layout().addWidget(self.qle_choice_lab_mod)
        self.layout().addWidget(self.btn_choice_labm)
        self.layout().addWidget(self.btn_run)


    def callback_option(self, value):
        """ This function is responsible of showing and hidding the widgets.
        """ 

        if value =='contour suggestion':
            self.Label_choice_labr.hide()
            self.qle_choice_lab_raw.hide()
            self.Label_choice_labm.hide()
            self.qle_choice_lab_mod.hide()
            self.btn_choice_labm.hide()

            self.Label_choice_img.move(10,40)            
            self.Label_choice_img.show()
            self.qle_choice_img.move(150,40)  
            self.qle_choice_img.show()  
            self.btn_choice_img.move(360,40)  
            self.btn_choice_img.show()  
            self.spbox_thresh.move(150,70) 
            self.spbox_thresh.setVisible(True)
            self.Label_thresh.move(10,70)
            self.Label_thresh.show()

        else:
            self.Label_choice_img.hide()
            self.qle_choice_img.hide()
            self.btn_choice_img.hide() 
            self.spbox_thresh.setHidden(True)
            self.Label_thresh.hide()
                           
            self.Label_choice_labr.move(10,40)
            self.Label_choice_labr.show()
            self.qle_choice_lab_raw.move(150,40)  
            self.qle_choice_lab_raw.show()  
            self.Label_choice_labm.move(10,70)
            self.Label_choice_labm.show()
            self.qle_choice_lab_mod.move(150,70)  
            self.qle_choice_lab_mod.show()  
            self.btn_choice_labm.move(360,55)  
            self.btn_choice_labm.show()  

    def callback_image_choice(self):
        self.qle_choice_img.clear()
        layers = list(self.viewer.layers)
        for l in layers:
            if isinstance(l, napari.layers.Image):
                self.qle_choice_img.addItem(l.name)

    def callback_label_choice(self):
        self.qle_choice_lab_raw.clear()
        self.qle_choice_lab_mod.clear()
        layers = list(self.viewer.layers)
        for l in layers:
            if isinstance(l, napari.layers.Labels):
                self.qle_choice_lab_mod.addItem(l.name)
                self.qle_choice_lab_raw.addItem(l.name)

    def callback_run(self):
        self.threshold = self.spbox_thresh.value()
        self.option = self.qle_option.currentText()
        self.image_name = self.qle_choice_img.currentText()
        self.lab_mod_name = self.qle_choice_lab_mod.currentText()
        self.lab_raw_name = self.qle_choice_lab_raw.currentText()
        layers = list(self.viewer.layers)
        for l in layers:
            if isinstance(l, napari.layers.Image):
                if l.name == self.image_name:
                    self.image_data = l.data
            elif isinstance(l, napari.layers.Labels):
                if l.name == self.lab_mod_name:
                    self.lab_mod_data = l.data
                if l.name == self.lab_raw_name:
                    self.lab_raw_data = l.data
         
        if self.option == 'contour suggestion':
            print('contour_suggestion',self.option)
            region_growing(self.viewer, self.image_data,self.threshold)
        else:
            print(self.option)
            assemble_labels (self.viewer,self.lab_raw_data,self.lab_mod_data)


class save_load (QMainWindow):
    """ This class widget allows the user to save or load an image.

    This widget gives the user the ability to:
    - Choose to save or load an image.
    - Choose the path where to save or load from an image.
    - Choose the layer type ('Labels' or 'Image') for the loaded image   

    """
    def __init__(self, napari_viewer, parent=None):
        """ QWidget.__init__ method.

        Parameters
        ----------
        napari_viewer : instance
            Access to napari viewer in order to add widgets.
        parent : class
            Parent class (QWidget)
        """
        super().__init__(parent)
        
        self.viewer = napari_viewer
        self.layers = napari.layers
        self.choice = 'save'

        # Choose either to save or load an image
        self.Label_choice = QLabel(self)
        self.Label_choice.setText('Choice')
        self.qle_choice = QComboBox(self)
        self.qle_choice.setFixedSize(200,25)
        self.qle_choice.addItem('save')
        self.qle_choice.addItem('load')
        self.qle_choice.currentTextChanged.connect(self.callback_option) 
        self.Label_choice.move(10,10)
        self.qle_choice.move(150,10)  

        # pick a path                               
        self.Label_path = QLabel(self)
        self.Label_path.setText('Path')
        self.btn_path = QPushButton('dir', self)
        self.btn_path.setFixedSize(200,25)
        self.Label_path.move(10,40)
        self.btn_path.move(150,40)  
        self.btn_path.clicked.connect(self.open_folder_in)        

        #Choose the layer type of the image to be loaded
        self.Label_layer_type = QLabel(self)
        self.Label_layer_type.setText('Layer_type')
        self.qle_layer_type = QComboBox(self)
        self.qle_layer_type.setFixedSize(200,25)
        self.qle_layer_type.addItem('label')
        self.qle_layer_type.addItem('Image')
        self.Label_layer_type.move(10,70)
        self.qle_layer_type.move(150,70)
        self.qle_layer_type.hide()
        self.Label_layer_type.hide()    


        # Execution Button 
        self.btn_save_load = QPushButton('Save_Load',self)
        self.btn_save_load.setFixedSize(200,25)
        self.btn_save_load.move(75,110)          
        self.btn_save_load.clicked.connect(self.callback_save_load)
    def callback_option(self,value):
        if value == 'save':
            self.qle_layer_type.hide()
            self.Label_layer_type.hide()    
        else:
            self.qle_layer_type.show()
            self.Label_layer_type.show()    

    def callback_save_load(self):
        v1 = self.qle_choice.currentText()
        self.choice = v1
        v2 = self.qle_layer_type.currentText()
        self.layer_type = v2
        if self.choice == "save":
            save_as_tiff(self.viewer, self.path)
        else:
            load_images(self.viewer, self.path, self.layer_type)

    def open_folder_in(self):
        v1 = self.qle_choice.currentText()
        self.choice = v1
        if self.choice == 'save':
            dialog = QFileDialog()
            dir = dialog.getExistingDirectory(self, 'Select an awesome directory')
            self.path= dir
        else:
            dialog = QFileDialog()
            dialog.setFileMode(QFileDialog.ExistingFiles)
            self.path= dialog.getOpenFileNames(self, filter='')
            print(self.path)		
            #self.path= QFileDialog.getOpenFileName()[0]


class Cellpose_Window_param(QWidget):
    """ This class widget  allows the user to set the parameters for Cellpose segmentation.

    This widget gives the user the ability to set the model parameters:
    """

    def __init__(self):
        super().__init__()
                    
        # model parameters            
        self.pretrained_pkl = []
        self.gpu = False
        self.model_type = 'None'
        self.diam_mean = 27
        #model evaluation parameters
        self.channels = [0, 0]
        self.flow_threshold = 0.4
        self.cellprob_threshold = 0.0
        self.min_size = 15

        self.Label_title = QLabel(self)
        self.Label_title.setText('model parameters')
        self.Label_title.setFixedSize(200,25)
        self.Label_title.move(10,5)

        self.Label_gpu = QLabel(self)
        self.Label_gpu.setText('GPU:')
        self.cbox_gpu = QComboBox(self)
        self.cbox_gpu.addItem('False')
        self.cbox_gpu.addItem('True')
        self.cbox_gpu.setFixedSize(200,25)
        self.Label_gpu.move(10,45)
        self.cbox_gpu.move(10,75)       

        self.Label_model_type = QLabel(self)
        self.Label_model_type.setText('model_type:')
        self.cbox_model_type = QComboBox(self)
        self.cbox_model_type.addItem('None')
        self.cbox_model_type.addItem('nuclei')
        self.cbox_model_type.addItem('cyto')
        self.cbox_model_type.setFixedSize(200,25)
        self.Label_model_type.move(10,105)
        self.cbox_model_type.move(10,135)       

        self.Label_diam_mean = QLabel(self)
        self.Label_diam_mean.setText('diam_mean:')
        self.qle_diam_mean = QLineEdit(self)
        self.qle_diam_mean.setPlaceholderText('Default: 27')
        self.qle_diam_mean.setFixedSize(200,25)
        self.Label_diam_mean.move(10,165)    
        self.qle_diam_mean.move(10,195)

        self.btn_pretrained = QPushButton('pretrained_pkl',self)
        self.btn_pretrained.setFixedSize(150,25)
        self.btn_pretrained.move(10,225)          
        self.btn_pretrained.clicked.connect(self.open_folder_in)        


        self.Label_title = QLabel(self)
        self.Label_title.setText('model evaluation parameters')
        self.Label_title.setFixedSize(300,25)
        self.Label_title.move(10,265)

        self.Label_channels = QLabel(self)
        self.Label_channels.setText('channels:')
        self.qle_channels = QLineEdit(self)
        self.qle_channels.setPlaceholderText('Default: [0, 0]')
        self.qle_channels.setFixedSize(200,25)
        self.Label_channels.move(10,305)
        self.qle_channels.move(10,335)       

        self.Label_flow_thresh = QLabel(self)
        self.Label_flow_thresh.setText('flow_threshold:')
        self.qle_flow_thresh = QLineEdit(self)
        self.qle_flow_thresh.setPlaceholderText('Default: 0.4')
        self.qle_flow_thresh.setFixedSize(200,25)
        self.Label_flow_thresh.move(10,365)
        self.qle_flow_thresh.move(10,395)       

        self.Label_cellprob_thresh = QLabel(self)
        self.Label_cellprob_thresh.setText('cellprob_threshold:')
        self.qle_cellprob_thresh = QLineEdit(self)
        self.qle_cellprob_thresh.setPlaceholderText('Default: 0.0')
        self.qle_cellprob_thresh.setFixedSize(200,25)
        self.Label_cellprob_thresh.move(10,425)
        self.qle_cellprob_thresh.move(10,455)
        
        self.Label_min_size = QLabel(self)
        self.Label_min_size.setText('min_size:')
        self.qle_min_size = QLineEdit(self)
        self.qle_min_size.setPlaceholderText('Default: 15')
        self.qle_min_size.setFixedSize(200,25)
        self.Label_min_size.move(10,485)
        self.qle_min_size.move(10,515)       

        self.btn_in_param = QPushButton('Input Parameters',self)
        self.btn_in_param.setFixedSize(150,25)
        self.btn_in_param.move(10,560)          
        self.btn_in_param.clicked.connect(self.button_click)
        

    def button_click(self):

        #model parameters
        v1 = self.cbox_gpu.currentText()
        if v1 == 'False':
            self.gpu = False
        else:
            self.gpu = True    
        v2 = self.cbox_model_type.currentText()
        if v2 == 'None':
            self.model_type = None
        else:
            self.model_type = v2    
        v3 = self.qle_diam_mean.text()
        if bool(v3):
            self.diam_mean = literal_eval(v3)

        #model evaluation parameters                
        v5 = self.qle_channels.text()
        if bool(v5):
            self.channels = literal_eval(v5) 
        v6 = self.qle_flow_thresh.text()
        if bool(v6):
            self.flow_threshold = literal_eval(v6) 
        v7 = self.qle_cellprob_thresh.text()
        if bool(v7):
            self.cellprob_threshold = literal_eval(v7) 
        v8 = self.qle_min_size.text()
        if bool(v8):
            self.min_size = literal_eval(v8) 

    def open_folder_in(self):
        dialog = QFileDialog()
        self.pretrained_pkl = QFileDialog.getOpenFileName()[0]


class training (QMainWindow):
    """ This class widget allows the user to train a model and to do prediction using a pretrained algorithm.

    This widget gives the user the ability to:
    - Choose the parameters for the trainig of the model.
    - Choose the parameters for the prediction of the output.
    """

    def __init__(self, napari_viewer, parent=None):
        """ QWidget.__init__ method.

        Parameters
        ----------
        napari_viewer : instance
            Access to napari viewer in order to add widgets.
        parent : class
            Parent class (QWidget)
        """    
        super().__init__(parent)
        self.window = Cellpose_Window_param()
        self.viewer = napari_viewer
        # self.window1 = AnotherWindow()
        #           
        self.lab_train = QLabel(self)
        self.lab_train.setText('train')
        self.lab_train.move(10,50)                       
        self.btn_ok = QPushButton('OK',self)
        self.btn_ok.setFixedSize(50,20)
        self.btn_ok.move(280,50) 
        self.btn_ok.clicked.connect(self.callback_train)   
        self.btn_param = QPushButton('param',self)
        self.btn_param.setFixedSize(50,20)
        self.btn_param.move(225,50)
        # self.slider.clicked.connect(self.img_mask_window)

        self.lab_eval = QLabel(self)
        self.lab_eval.setText('train')
        self.lab_eval.move(10,85)                       
        self.btn_ok1 = QPushButton('OK',self)
        self.btn_ok1.setFixedSize(50,20)
        self.btn_ok1.move(280,85) 
        self.btn_ok1.clicked.connect(self.callback_eval)   
        self.btn_param1 = QPushButton('param',self)
        self.btn_param1.setFixedSize(50,20)
        self.btn_param1.move(225,85)
        self.slider.clicked.connect(self.callback_param)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self.lab_train)
        self.layout().addWidget(self.btn_ok)
        self.layout().addWidget(self.btn_param)
        self.layout().addWidget(self.lab_eval)
        self.layout().addWidget(self.btn_ok1)
        self.layout().addWidget(self.btn_param1)

    def callback_param(self, checked):
        if self.window.isVisible():
            self.window.hide()
        else:
            self.window.show()
        
    def callback_train(self):
        selected_layers = list(self.viewer.layers.selection)
        for l in selected_layers:
            if isinstance(l, napari.layers.Image):
                image = l.data
                image_name = l.name
            else:
                mask = l.data
                mask_name = l.name
        # path = '/home/slimane/Desktop/all/Big_Annotator/bigannotator/NEW/Ubuntu_BigAnnot/cell_pose/test_images/image/save_model/models/cellpose_residual_on_style_on_concatenation_off__2021_07_19_23_48_24/archive'
        input_images ='/home/slimane/Desktop/all/Big_Annotator/bigannotator/NEW/Ubuntu_BigAnnot/cell_pose/test_images/image/train1/'
        save_path = '/home/slimane/Desktop/all/design project/slimane_lio/cellpose_training/model weights/'        

        model = CellposeModel(gpu=False, pretrained_model=None, 
                model_type='nuclei', torch=True, diam_mean=None, net_avg=True, 
                device=None,residual_on=True, style_on=True, concatenation=False, nchan=2)        
        # image_list = []
        # for filename in glob.glob(input_images + '*.png'): #assuming gif
        #     im=imread(filename)
        #     image_list.append(filename)
        # l = int(len(image_list)/2)
        # X = np.empty((l,224,224))
        # Y = np.empty((l,224,224))
        # j=0
        # k=0
        # for i in range(l):
        #     for filename in (image_list):
        #         im=imread(filename) 
        #         if int(filename.split('/')[-1][:3]) == i:     
        #             if filename[-9:-4]!='masks':    
        #                 print('f',int(filename.split('/')[-1][:3]),'i',i,'filename[-9:-4]',filename[-9:-4])
        #                 X[j] = im
        #                 j+=1
        #             else:
        #                 print('f2',int(filename.split('/')[-1][:3]),'i2',i,'filename[-9:-4]',filename[-9:-4])
        #                 Y[k] = im
        #                 k+=1             

        # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.40, random_state=42)
        # X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.50, random_state=42)
        path1 = '/home/slimane/Desktop/all/design project/slimane_lio/cellpose_training/input_images'
        # import os
        # if not os.path.exists(path1):
        #         os.makedirs(path1)
        # train_files=[]
        # test_files=[]
        # val_files=[]
        # for i in range(len(X_train)):        
        #     tifffile.imwrite(path1+'/train'+'/THG_'+format(i,'03d')+'_img.tif',X_train[i])
        #     tifffile.imwrite(path1+'/train'+'/THG_'+format(i,'03d')+'_mask.tif',y_train[i])
        #     train_files.append(path1+'/train'+'/THG_'+format(i,'03d'))

        # for i in range(len(X_test)):        
        #     tifffile.imwrite(path1+'/test'+'/THG_'+format(i,'03d')+'_img.tif',X_test[i])
        #     tifffile.imwrite(path1+'/test'+'/THG_'+format(i,'03d')+'_mask.tif',y_test[i])
        #     test_files.append(path1+'/test'+'/THG_'+format(i,'03d'))

        # for i in range(len(X_val)):        
        #     tifffile.imwrite(path1+'/val'+'/THG_'+format(i,'03d')+'_img.tif',X_val[i])
        #     tifffile.imwrite(path1+'/val'+'/THG_'+format(i,'03d')+'_mask.tif',y_val[i])
        #     val_files.append(path1+'/val'+'/THG_'+format(i,'03d'))

        flow_train = labels_to_flows(y_train.astype(int), files=train_files)
        flow_val = labels_to_flows(y_val.astype(int), files=test_files)
        flow_test = labels_to_flows(y_test.astype(int), files=val_files)
        model.train(list(X_train), list(y_train.astype(int)), train_files=train_files, 
                    test_data=list(X_val), test_labels=list(y_val.astype(int)), 
                    test_files=val_files, channels= [0,0], 
                    normalize=True, pretrained_model=None, 
                    save_path=save_path, save_every=1, 
                    learning_rate=0.2, n_epochs=4, 
                    momentum=0.9, weight_decay=1e-05, 
                    batch_size=1, rescale=True)


        # self.viewer.add_image(flow, name = 'flow_image')
        # self.viewer.add_image(y_train.astype(int), name = 'flow_image')

    def callback_eval(self):
        selected_layers = list(self.viewer.layers.selection)
        for l in selected_layers:
            if isinstance(l, napari.layers.Image):
                image = l.data
                image_name = l.name
        image = image.astype(np.uint16)
        from cellpose.models import CellposeModel
        path = '/home/slimane/Desktop/all/Big_Annotator/bigannotator/NEW/Ubuntu_BigAnnot/cell_pose/test_images/image/save_model/models/cellpose_residual_on_style_on_concatenation_off__2021_07_19_23_48_24/archive'
        model1 = CellposeModel(gpu=False, pretrained_model=path+'/data.pkl', 
                model_type='nuclei', torch=True, diam_mean=36.4, net_avg=True, 
                device=None,residual_on=True, style_on=True, concatenation=False, nchan=2)        
        masks,flows,styles = model1.eval(image, batch_size=1, channels=[0,0], channel_axis=None, 
                                        z_axis=None, normalize=True, invert=False, rescale=None, 
                                        diameter=36.4, do_3D=False, anisotropy=None, 
                                        net_avg=True, augment=False, tile=True, tile_overlap=0.1, 
                                        resample=True, interp=True, flow_threshold=0.4, 
                                        cellprob_threshold=0.0, compute_masks=True, min_size=-1, 
                                        stitch_threshold=0.0, progress=None)
        
        # import tifffile
        # tifffile.imwrite(path1+'/THG_'+format(0,'03d')+'_img.tif',X_train)
        # tifffile.imwrite(path1+'/THG_'+format(0,'03d')+'_mask.tif',y_train)
        # from cellpose.dynamics import labels_to_flows
        # flow = labels_to_flows(y_train.astype(int), files=None)
        # tifffile.imsave(path1+'/THG_'+format(0,'03d')+'_flows.tif', flow)
        self.viewer.add_image(flows[2], name = 'flows2')
        self.viewer.add_image(flows[0], name = 'flows0')
        self.viewer.add_image(flows[1], name = 'flows1')
        self.viewer.add_image(flows[3], name = 'flows3')

        self.viewer.add_labels(masks.astype(int), name = 'mask')

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    """napari plugins GUI hook.

    Returns
    -------
    list(callable)
        A “callable” in this context is a class or function that, 
        when called, returns an instance of either a QWidget or a FunctionGui.
    Note
    ----
    You can return either a single widget, or a sequence of widgets  
    """
    return [ImagePreprocessing, segmentation, save_load, contour_assist,training,Extract_imagette,Image_reconstruction,Image_classification]
