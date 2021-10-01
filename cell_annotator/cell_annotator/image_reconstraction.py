 #Selection of version of tensorflow in colab
#%tensorflow_version 1.x

# library import
# import keras
from tensorflow.compat.v1  import  roll
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import Input, Add, Lambda, Subtract,Layer
from tensorflow.keras.optimizers import *
from math import pi as pi
from tensorflow.keras import backend as K
from math import pi as pi
import matplotlib.pyplot as plt
import math
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from skimage.transform import rescale



from tensorflow.keras.constraints import Constraint
from mpl_toolkits.axes_grid1 import make_axes_locatable

from natsort import natsorted
import glob
import imageio
from skimage.io         import imread, imsave
from skimage.measure    import regionprops
from skimage.io         import imsave
print('finished importing')



# Functions

def spiral_kxky(filename, ledNum):
    kxky = [[], []]
    with open(filename, 'r') as file:
        for line in file:
            for j, value in enumerate(line.split(",")):
                kxky[j].append(np.float(value))
    kxky = np.asarray(kxky)
    kxky = kxky.T
    return kxky[:ledNum, :]


def show_result(viewer, dir, model, imSize, show=0, noShow=10,size=10):
    w_conv_Or = model.get_layer('O_FTr').get_weights()
    w_conv_Oi = model.get_layer('O_FTi').get_weights()
    w_conv_Or_array = np.asarray(w_conv_Or)
    w_conv_Oi_array = np.asarray(w_conv_Oi)
    c_real = w_conv_Or_array[0, :, :, 0].reshape((imSize, imSize))
    c_imag = w_conv_Oi_array[0, :, :, 0].reshape((imSize, imSize))
    
    c_complex = c_real + 1j * c_imag
    c_abs = np.abs(c_complex)
    c_phase = np.angle(c_complex+pi)
    im_spatial = np.abs(np.fft.ifft2(np.fft.ifftshift(c_complex)))
    im_phase = np.angle(np.fft.ifft2(np.fft.ifftshift(c_complex)))
    
    if show:
        viewer.add_image(np.log(c_abs+1), name='recover (abs)')
        viewer.add_image(np.log(c_abs), name='recover (abs)1')
        viewer.add_image(c_abs, name='(abs)')
        viewer.add_image(im_phase, name='recover (phase)')
        viewer.add_image(im_spatial, name='recover (spatial)')
        imsave(dir+ 'recovered_phase'+'.tif', im_phase)
        imsave(dir+ 'recovered_spatial'+'.tif', im_spatial)
        imsave(dir+ 'recovered_FT'+'.tif', np.log(c_abs+1))
        # plt.figure(figsize=(size,size))
        # plt.subplot(233),plt.imshow(np.log(c_abs[noShow:imSize-noShow, noShow:imSize-noShow]+1), cmap='gray'),plt.title('recover (abs)')
        # ax = plt.subplot(232)
        # plot = plt.imshow(im_phase[noShow:imSize-noShow, noShow:imSize-noShow], cmap='gray')
        # divider = make_axes_locatable(ax)
        # plt.title('recover (phase)')
        # cax = divider.append_axes("right", size="5%", pad=0.02)
        # plt.colorbar(plot,cax=cax)
        # plt.subplot(231),plt.imshow(im_spatial[noShow:imSize-noShow, noShow:imSize-noShow], cmap='gray'),plt.title('recover FT')
        # plt.show()
        # plt.show()
        
    return c_complex


class MyLayer( Layer):
    def __init__(self, output_dims,imSize, **kwargs):
        self.output_dims = output_dims
        self.imSize = imSize

        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=self.output_dims,
                                      initializer='ones',
                                      trainable=True)

        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        a = tf.keras.backend.reshape(x, shape=(-1,self.imSize,self.imSize,1))
        return tf.multiply(a,self.kernel)

    def compute_output_shape(self, input_shape):
        return (self.output_dims)


class take_one( Layer):
    def __init__(self, output_dims, **kwargs):
        self.output_dims = output_dims

        super(take_one, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=self.output_dims,
                                      initializer='ones',
                                      trainable=False)

        super(take_one, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        b = tf.keras.backend.reshape(x, shape=(1,1))
        a = tf.keras.backend.cast(b, dtype='int32')
        return a[:,0]*1

    def compute_output_shape(self, input_shape):
        return (self.output_dims)


class Between(Constraint):
    def __init__(self, min_value, max_value):
        self.min_value =  min_value
        self.max_value = max_value

    def __call__(self, w):        
        return K.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}

class ConvexCombination(Layer):
    def __init__(self, **kwargs):
        super(ConvexCombination, self).__init__(**kwargs)

    def build(self, input_shape):
        self.lambd2 = self.add_weight(name='lambda2',
                                     shape=(10,1),  # Adding one dimension for broadcasting
                                     initializer='ones',  # Try also 'ones' and 'uniform'
                                     trainable=True,constraint =Between(-1.,1.) )
        super(ConvexCombination, self).build(input_shape)

    def call(self, x):
        # x is a list of two tensors with shape=(batch_size, H, T)
        h1,h2,h3,h4,h5,h6,h7,h8,h9,h10 = x
        a= self.lambd2[0,0]
        b= self.lambd2[1,0]
        c= self.lambd2[2,0]
        d= self.lambd2[3,0]
        e= self.lambd2[4,0]
        f= self.lambd2[5,0]
        g= self.lambd2[6,0]
        h= self.lambd2[7,0]
        i= self.lambd2[8,0]
        j= self.lambd2[9,0]
        # k= self.lambd2[10,0]
        # l= self.lambd2[11,0]

        new_ctf = a*h1 + b*h2 +  c*h3 + d*h4 + e*h5 + f*h6 + g*h7 + h*h8+i*h9 +j*h10#+ k*h11 +l*h12
        return new_ctf

    def compute_output_shape(self, input_shape):
        return input_shape[0]



# Generate CTF

def Generate_CTF(psize,imSize,NA,k0,imCenter):
  dkxy = 2*pi/(psize*imSize)
  cutoffFrequency = (NA * k0 / dkxy)
  center = [imCenter, imCenter]
  kYY, kXX = np.ogrid[:imSize, :imSize]
  CTF = np.sqrt((kXX - center[0]) ** 2 + (kYY - center[1]) ** 2) <= cutoffFrequency
  CTF = CTF.astype(float)
  return CTF, dkxy
# Show CTF
# plt.figure(figsize=(15,15))
# plt.xticks(np.arange(0,crop,20))
# plt.subplot(1, 3, 3),plt.imshow(CTF[:,:], cmap='gray'),plt.title('CTF')
# plt.show()

# print(cutoffFrequency)


from LightPipes import *
import matplotlib.pyplot as plt
import math




def generate_Zernike(wavelength,NA, k0, dkxy, imSize,psize):
  cutoffFrequency = (NA * k0 / dkxy)
  size=cutoffFrequency
  N=imSize
  A=wavelength/(N/2*math.pi)
  poly = np.ndarray([21,imSize,imSize],dtype=np.longdouble)
# plt.figure(figsize=(15,8)) 
  for Noll in range (1,15):
      (nz,mz)=noll_to_zern(Noll)
      S=ZernikeName(Noll)
      F=Begin(N,psize,N)
      F=Zernike(nz,mz,size,A,F)
      F=CircAperture(size,0,0,F)
      Phi=Phase(F)
      Z = Phi
      poly[Noll-1] = Z
  del Z,Phi,S,F    
  return poly
      # ax1 = plt.subplot(3,7,Noll)
      # plot = plt.imshow(np.real(Z), cmap='gray')
      # plt.colorbar(plot,ax=ax1)
      # s=repr(Noll) + '  ' + ' $Z^{'+repr(mz)+'}_{'+repr(nz)+'}$' + '\n' + S
      # plt.title(s, fontsize=9);plt.axis('off')

#     ax1 = plt.subplot(3,7,Noll)
#     plot = plt.imshow(poly[Noll-1], cmap='gray')
#     plt.colorbar(plot,ax=ax1)
#     s=repr(Noll) + '  ' + ' $Z^{'+repr(mz)+'}_{'+repr(nz)+'}$' + '\n' + S#
#     plt.title(s, fontsize=9);plt.axis('off')
# plt.show()

def create_Z_inputs(poly, imSize, arraysize):
  

  z1_input = np.ndarray((int(arraysize ** 2), imSize*imSize, 1),dtype=np.float32) # input CTF
  z2_input = np.ndarray((int(arraysize ** 2), imSize*imSize, 1),dtype=np.float32) # input CTF
  z3_input = np.ndarray((int(arraysize ** 2), imSize*imSize, 1),dtype=np.float32) # input CTF
  z4_input = np.ndarray((int(arraysize ** 2), imSize*imSize, 1),dtype=np.float32) # input CTF
  z5_input = np.ndarray((int(arraysize ** 2), imSize*imSize, 1),dtype=np.float32) # input CTF
  z6_input = np.ndarray((int(arraysize ** 2), imSize*imSize, 1),dtype=np.float32) # input CTF
  z7_input = np.ndarray((int(arraysize ** 2), imSize*imSize, 1),dtype=np.float32) # input CTF
  z8_input = np.ndarray((int(arraysize ** 2), imSize*imSize, 1),dtype=np.float32) # input CTF
  z9_input = np.ndarray((int(arraysize ** 2), imSize*imSize, 1),dtype=np.float32) # input CTF
  z10_input = np.ndarray((int(arraysize ** 2), imSize*imSize, 1),dtype=np.float32) # input CTF

  for i in range(int(arraysize ** 2)):
      z1_input[i, :, 0]  = np.reshape(poly[0],(imSize*imSize))
      z2_input[i, :, 0]  = np.reshape(poly[1],(imSize*imSize))
      z3_input[i, :, 0]  = np.reshape(poly[2],(imSize*imSize))
      z4_input[i, :, 0]  = np.reshape(poly[3],(imSize*imSize))
      z5_input[i, :, 0]  = np.reshape(poly[4],(imSize*imSize))
      z6_input[i, :, 0]  = np.reshape(poly[5],(imSize*imSize)) 
      z7_input[i, :, 0]  = np.reshape(poly[6],(imSize*imSize)) 
      z8_input[i, :, 0]  = np.reshape(poly[7],(imSize*imSize)) 
      z9_input[i, :, 0]  = np.reshape(poly[8],(imSize*imSize)) 
      z10_input[i, :, 0]  = np.reshape(poly[9],(imSize*imSize))

  del poly

  return z1_input, z2_input, z3_input, z4_input, z5_input, z6_input, z7_input, z8_input, z9_input, z10_input



def create_models(imSize, crop,index_downSample):
  center = imSize/2

  input_all = Input((imSize*imSize+2, 1),dtype='complex64', name='input_all')  # CTF
  input_measurement = Input((crop, crop, 1), name='input_measurement')  # measurement
  nule = Input((imSize,imSize, 1),dtype='float32', name='input_nule')  # CTF tf.ones(tf.shape(input_CTF), tf.float32)

  hx = int(center-crop/2)
  hy = int(center +crop/2)

  input_CTF = Lambda(lambda x: x[:,:imSize*imSize,:],name='input_CTF')(input_all)
  input_kx = Lambda(lambda x:  tf.real(x[:,imSize*imSize:imSize*imSize+1,:]),name='input_kx')(input_all)
  input_ky = Lambda(lambda x: tf.real(x[:,imSize*imSize+1:imSize*imSize+2,:]),name='input_ky')(input_all)

  z1 = Input((imSize*imSize, 1),dtype='float', name='z1')  # CTF
  z2 = Input((imSize*imSize, 1),dtype='float', name='z2')  # CTF
  z3 = Input((imSize*imSize, 1),dtype='float', name='z3')  # CTF
  z4 = Input((imSize*imSize, 1),dtype='float', name='z4')  # CTF
  z5 = Input((imSize*imSize, 1),dtype='float', name='z5')  # CTF
  z6 = Input((imSize*imSize, 1),dtype='float', name='z6')  # CTF
  z7 = Input((imSize*imSize, 1),dtype='float', name='z7')  # CTF
  z8 = Input((imSize*imSize, 1),dtype='float', name='z8')  # CTF
  z9 = Input((imSize*imSize, 1),dtype='float', name='z9')  # CTF
  z10 = Input((imSize*imSize, 1),dtype='float', name='z10')  # CTF
  z11 = Input((imSize*imSize, 1),dtype='float', name='z11')  # CTF
  z12 = Input((imSize*imSize, 1),dtype='float', name='z12')  # CTF


  kx_i = take_one((1,),input_shape = (1,1))(input_kx)
  ky_i = take_one((1,),input_shape = (1,1))(input_ky)

  # define O (FT)
  O_FTr = MyLayer((imSize, imSize, 1),imSize=imSize, input_shape= (imSize* imSize, 1), name='O_FTr')
  O_FTi = MyLayer((imSize, imSize, 1),imSize=imSize, input_shape= (imSize* imSize, 1), name='O_FTi')

  # define P
  P_r = MyLayer((imSize, imSize, 1),imSize=imSize, input_shape= (imSize* imSize, 1), name='P_r')
  P_i = MyLayer((imSize, imSize, 1),imSize=imSize, input_shape= (imSize* imSize, 1), name='P_i')

  # CTF * O (FT)
  CTFr = Lambda(lambda x: tf.real(x))(input_CTF)
  CTFi = Lambda(lambda x: tf.imag(x))(input_CTF)


  Comb = ConvexCombination(name='inter_z1')([z1,z2,z3,z4,z5,z6,z7,z8,z9,z10])

  Pupil = Lambda(lambda x: tf.exp(1j*tf.cast(x[0],tf.complex64))*x[1],name='Pupil')([Comb,input_CTF])


  Pupil_r = Lambda(lambda x: tf.real(x))(Pupil)
  Pupil_i = Lambda(lambda x: tf.imag(x))(Pupil)


  P_cr = P_r(Pupil_r)
  P_ci = P_i(Pupil_i)

  Or = O_FTr(nule)
  Oi = O_FTi(nule)


  Or_d =  Lambda(lambda x: roll(x[0],shift=[x[1][0],x[2][0]],axis=[1,2]),name='Roll_Or')([Or,kx_i,ky_i])  
  Oi_d =  Lambda(lambda x: roll(x[0],shift=[x[1][0],x[2][0]],axis=[1,2]),name='Roll_Oi')([Oi,kx_i,ky_i])


  CrOr_c = Lambda(lambda x: tf.multiply(x[0],x[1]))([P_cr,Or_d])
  CiOi_c = Lambda(lambda x: tf.multiply(x[0],x[1]))([P_ci,Oi_d])
  CrOi_c = Lambda(lambda x: tf.multiply(x[0],x[1]))([P_cr,Oi_d])
  CiOr_c = Lambda(lambda x: tf.multiply(x[0],x[1]))([P_ci,Or_d])

  CrOr = Lambda(lambda x: tf.image.central_crop(x, 1/index_downSample ),name='Crop_CrOr')(CrOr_c)
  CiOi = Lambda(lambda x: tf.image.central_crop(x, 1/index_downSample ),name='Crop_CiOi')(CiOi_c)
  CrOi = Lambda(lambda x: tf.image.central_crop(x, 1/index_downSample ),name='Crop_CrOi')(CrOi_c)
  CiOr = Lambda(lambda x: tf.image.central_crop(x, 1/index_downSample ),name='Crop_CiOr')(CiOr_c)

  # generate low resolution image (FT)
  lowFT_r = Subtract(name='lowFT_r')([CrOr, CiOi])
  lowFT_i = Add(name='lowFT_i')([CrOi, CiOr])

  lowFT = Lambda(lambda x: tf.cast(x[0], tf.complex64) + 1j * tf.cast(x[1], tf.complex64),name='lowFT')([lowFT_r, lowFT_i])
  # do ifft
  im_iFT = Lambda(lambda x: tf.ifft3d(tf.manip.roll(tf.cast(x, tf.complex64),[int(imSize/2), int(imSize/2)], axis=[0, 1])),name='low_iFT')(lowFT)

  image = Lambda(lambda x: tf.abs(x))(im_iFT)
  # keep angle, and use sqrt(I) to change the anplitude
  iFT_angle = Lambda(lambda x: tf.angle(tf.cast(x, tf.complex64)),name='low_iFT_angle')(im_iFT)

  sqrtI = Lambda(lambda x: tf.sqrt(x),name='input_amp')(input_measurement)

  output = Lambda(lambda x: tf.multiply(tf.cast(tf.divide(tf.reduce_sum(x[0]),tf.reduce_sum(x[1])), tf.float32),x[2]),output_shape=(crop,crop,1),name='out')([sqrtI, image,image])

  model = Model(inputs=[input_all,input_measurement,nule,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10], outputs=[output])
  #model.summary()

  return model

  #keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)

def updating_pos(arraysize,crop):

  kakr = spiral_kxky('/home/slimane/Desktop/all/design project/slimane_lio/Image reconstruction/training/pos_all_2.txt',int( arraysize ** 2)) # load kx, ky here
  #print('kxky shape:',kakr.shape)

  kx_i = np.ndarray((int(arraysize ** 2),1,1),dtype=np.float32) # input measurement
  ky_i = np.ndarray((int(arraysize ** 2),1,1),dtype=np.float32) # input measurement


  # Updating of positions according to errors found by the genetic algorithm
  kakr[1,0] = kakr[1,0] +14.76734925
  kakr[1,1] = kakr[1,1] + 0.59890099

  kakr[2,0] = kakr[2,0] +14.89285647
  kakr[2,1] = kakr[2,1] +0.92978567

  kakr[3,0] = kakr[3,0] + 14.10572814 
  kakr[3,1] = kakr[3,1] - 0.05350908

  kakr[4,0] = kakr[4,0] + 14.70121004
  kakr[4,1] = kakr[4,1] -0.96501889

  kakr[5,0] = kakr[5,0] +14.02508033
  kakr[5,1] = kakr[5,1] + 0.82702106

  kakr[6,0] = kakr[6,0] + 14.97020396
  kakr[6,1] = kakr[6,1] -  0.0542885

  kakr[7,0] = kakr[7,0] -14.96710334  
  kakr[7,1] = kakr[7,1] -0.77540696

  kakr[9,0] = kakr[9,0] + 3.36549871
  kakr[9,1] = kakr[9,1]  -0.40575659

  kakr[11,0] = kakr[11,0] + 6.61696441 
  kakr[11,1] = kakr[11,1] -0.65608742

  kakr[12,0] = kakr[12,0] + 6.93462461 
  kakr[12,1] = kakr[12,1] + 0.29434819

  kakr[13,0] = kakr[13,0] + 5.15290666 
  kakr[13,1] = kakr[13,1] + 0.62740655

  kakr[14,0] = kakr[14,0] + 4.9515296
  kakr[14,1] = kakr[14,1] + 0.46726247

  kakr[15,0] = kakr[15,0] + 0.88872929 
  kakr[15,1] = kakr[15,1] -0.00710684

  kakr[16,0] = kakr[16,0] + 3.60385903 
  kakr[16,1] = kakr[16,1] + 0.46693824

  kakr[18,0] = kakr[18,0] -14.55634447  
  kakr[18,1] = kakr[18,1] +  0.74065819

  for i in range(35):
    kx_i[i,0,0] = (kakr[i,0]*(crop/500))*np.cos(kakr[i,1])
    ky_i[i,0,0] = (kakr[i,0]*(crop/500))*np.sin(kakr[i,1])
    #print(kakr[i])
  return kx_i, ky_i  

#Function that uses the mask at the end of the FPM to save the cropped image
def extract_imagette(image, labeled,raw, coords_para, coords_distrac,cells_mean=60, size1=71, size2=71, 
                     travel_output=os.getcwd()+"_output/",tophat=False, zoro=False,mask=False):
  
    for region in regionprops(labeled):

        image_sortie=np.zeros([size1,size2], dtype='uint8')

        if region.equivalent_diameter > cells_mean*2/3:

            coords = region.coords
            
            #imagette image
            image_sortie[coords[:,0],coords[:,1]]=\
            image[coords[:,0],coords[:,1]]

            imsave(travel_output, image_sortie)



def reconstruction_function(input_image_dir ,output_image_dir, save_path,arraysize,imSize,crop,index_downSample,wavelength,NA,k0,imCenter,psize):
  # Generate CTFs
  kx_input = np.ndarray((int(arraysize ** 2),1,1),dtype=np.float32) # input measurement
  ky_input = np.ndarray((int(arraysize ** 2),1,1),dtype=np.float32) # input measurement

  # Call all function necessary
  CTF, dkxy = Generate_CTF(psize=psize,imSize=imSize,NA=NA,k0=k0,imCenter=imCenter)
  kx_i, ky_i = updating_pos(arraysize=arraysize,crop=crop)
  poly = generate_Zernike(wavelength=wavelength,NA=NA, k0=k0, dkxy=dkxy, imSize=imSize,psize=psize)
  z1_input, z2_input, z3_input, z4_input, z5_input, z6_input, z7_input, z8_input, z9_input, z10_input = create_Z_inputs(poly=poly, imSize=imSize, arraysize=arraysize)

  # Execution of neural network to do the reconstruction FPM and save the image
  import time
  original =np.ndarray((35, crop, crop))

  # input data
  login = input_image_dir

  print('list_dir',os.listdir(login))

  # choice witch directory you want work
  log = os.listdir(login)[0] 
# for log in list_log:
  print('log',log)
  travel = login+'/'

  infected = []
  # healthy =[]
    # output data
  travel_output = output_image_dir +'/'+log+'/'

  if not os.path.exists(travel_output):
     os.makedirs(travel_output)

  for arch in os.listdir(travel):
    # if 'infected' in arch:
    infected.append(arch)
    print('infected',infected)
  #   if 'healthy' in arch:
  #     healthy.append(arch)

  count = 0 
  print('len(infected)',len(infected))
  Loss=np.zeros((len(infected),20))
  for ch in infected:
      print('ch',ch)
      start = time.time()
      tf.keras.backend.clear_session()
      title1 ='_' + ch +'_0.png'
      title2 ='_' + ch +'_1.png'
      i = 0
      # celule = travel + ch + '/'
      # for image_path in natsorted(glob.glob(celule+'*.png')):
      from skimage.io import imread, imshow
      for image_path in natsorted(glob.glob(travel+'*.')):
        image = imread(image_path)
        print('image.shape',image.shape)
        for i in range(35):
          original[i] = image[i]
        # if 'mask' not in image_path:
        #   original[i] = cv2.fastNlMeansDenoising(image,None,3,7,15)  
        #   i = i+1
        # else:
        #   mask = label(resize(image, (image.shape[0] *2, image.shape[1]*2),anti_aliasing=True))[0]
      divs = original.copy()
            
      cell = divs

      # Generate CTFs
      imgs_train_input1 = np.ndarray((int(arraysize ** 2), imSize, imSize, 1),dtype=np.complex64) # input CTF
      imgs_train_input2 = np.ndarray((int(arraysize ** 2), crop, crop, 1)) # input CTF
      imgs_train_input4 = np.ndarray((int(arraysize ** 2), imSize, imSize, 1),dtype=np.float32) # input CTF

      for i in range(int(arraysize ** 2)):
          imgs_train_input2[i, :, :, 0] = cell[i]
          imgs_train_input1[i, :, :, 0] = CTF.astype(np.complex64)
          imgs_train_input4[i, :, :, 0] = np.ones(( imSize, imSize),np.float32)

      imgs_train_input3 = np.reshape(imgs_train_input1,(int(arraysize ** 2),imSize* imSize, 1))

      imgs_train_input3 = np.concatenate((imgs_train_input3,kx_i,ky_i),axis=1)

      model = create_models()

      weight_or = np.ndarray((1, imSize, imSize, 1))
      weight_oi = np.ndarray((1, imSize, imSize, 1))

      weight_pr = np.ndarray((1, imSize, imSize, 1))
      weight_pi = np.ndarray((1, imSize, imSize, 1))

      weight_z1 = np.zeros((10, 1))

      imgs_test_predict = np.sqrt(imgs_train_input2)

      # set low res image FT as the initial weight
      imlowFT1 = np.fft.fftshift(np.fft.fft2(np.sqrt( rescale(imgs_train_input2[0, :, :, 0], index_downSample,anti_aliasing=False)))).astype(np.complex128)
      weight_or[0, :, :, 0] = np.real(imlowFT1)
      weight_oi[0, :, :, 0] = np.imag(imlowFT1)

      model.get_layer('O_FTr').set_weights(weight_or)
      model.get_layer('O_FTi').set_weights(weight_oi)
      model.get_layer('inter_z1').set_weights([weight_z1])

      model.get_layer('P_r').trainable = False
      model.get_layer('P_i').trainable = False
      model.get_layer('O_FTr').trainable = False
      model.get_layer('O_FTi').trainable = False
      model.get_layer('inter_z1').trainable = True

      # Trains the neural net only the layer of Zernike polynomials 

      # adam = Adagrad(learning_rate=0.01)
      def scheduler(epoch, learning_rate):
          if epoch > 0 and epoch%3 ==0:
            print('epoch',epoch,'learning_rate',learning_rate) 
            return learning_rate * tf.math.exp(-0.5)
          else:
            print('else_epoch',epoch,'else_learning_rate',learning_rate) 
            return learning_rate
      from tf.keras.optimizers import Adam
      from tf.keras.callbacks import LearningRateScheduler      
      callback = LearningRateScheduler(scheduler)
      opt1 = Adam(lr=0.01)
      model.compile(optimizer=opt1, loss="mse", metrics=["mae", "acc"])
      # model.compile(loss='mean_absolute_error', optimizer=adam)
      history = model.fit([imgs_train_input3, imgs_train_input2,imgs_train_input4,z1_input,z2_input,z3_input,z4_input,z5_input,z6_input,z7_input,z8_input,z9_input,z10_input],imgs_test_predict, batch_size=1, epochs=20,callbacks=[callback], verbose=1, shuffle=False)


      model.get_layer('O_FTr').trainable = True
      model.get_layer('O_FTi').trainable = True
      model.get_layer('inter_z1').trainable = False

      w_conv_Or = model.get_layer('O_FTr').get_weights()
      w_conv_Oi = model.get_layer('O_FTi').get_weights()
      w_conv_Or_array = np.asarray(w_conv_Or)
      w_conv_Oi_array = np.asarray(w_conv_Oi)
      c_real = w_conv_Or_array[0, :, :, 0].reshape((imSize, imSize))
      c_imag = w_conv_Oi_array[0, :, :, 0].reshape((imSize, imSize))

      weight_or[0, :, :, 0] = c_real
      weight_oi[0, :, :, 0] = c_imag 

      model.get_layer('O_FTr').set_weights(weight_or)
      model.get_layer('O_FTi').set_weights(weight_oi)

      # Train the complex object
      # adam = RMSprop(learning_rate=10)
      # model.compile(loss="mean_absolute_error", optimizer=adam)
      learning_rate = 10
      def scheduler(epoch, learning_rate):
          if epoch > 0 and epoch%3 ==0: 
            print('epoch',epoch,'learning_rate',learning_rate)
            return learning_rate * tf.math.exp(-0.5)
          else:
            return learning_rate
      callback = LearningRateScheduler(scheduler)
      opt2 = adam(lr=10)
      model.compile(optimizer=opt1, loss="mse", metrics=["mae", "acc"])
      model.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])

      history = model.fit([imgs_train_input3, imgs_train_input2,imgs_train_input4,z1_input,z2_input,z3_input,z4_input,z5_input,z6_input,z7_input,z8_input,z9_input,z10_input],imgs_test_predict, batch_size=1, epochs=100,callbacks=[callback], verbose=1, shuffle=False)
      Loss[count,:]=history.history['loss']

      # Takes the object and performs the inverse Fourier transform to get the intensity and phase image and save it to file
      w_conv_Or = model.get_layer('O_FTr').get_weights()
      w_conv_Oi = model.get_layer('O_FTi').get_weights()
      w_conv_Or_array = np.asarray(w_conv_Or)
      w_conv_Oi_array = np.asarray(w_conv_Oi)
      c_real = w_conv_Or_array[0, int(imSize/2)-int(crop/2):int(imSize/2)+int(crop/2), int(imSize/2)-int(crop/2):int(imSize/2)+int(crop/2), 0].reshape((crop, crop))
      c_imag = w_conv_Oi_array[0, int(imSize/2)-int(crop/2):int(imSize/2)+int(crop/2), int(imSize/2)-int(crop/2):int(imSize/2)+int(crop/2), 0].reshape((crop, crop))

      c_complex = c_real + 1j * c_imag
      c_abs = np.abs(c_complex)
      c_phase = np.angle(c_complex+pi)
      im_spatial = np.abs(np.fft.ifft2(np.fft.ifftshift(c_complex)))
      im_phase = np.angle(np.fft.ifft2(np.fft.ifftshift(c_complex)))
        
        # to save the image in intensity and phase
        # os.makedirs(travel_output)
      plt.figure(figsize=(10,10))
      plt.imshow(im_spatial,cmap='gray')
      plt.figure(figsize=(10,10))
      plt.imshow(im_phase,cmap='gray')

      # viewer.add_image(im_spatial,name='phase image')

      # viewer.add_image(im_spatial,name='spatial image')
      # viewer.add_image(im_spatial,name='phase image')
      model.save(save_path+'model.h5')
  #     imsave(travel_output+title1, im_spatial)
  #     imsave(travel_output+title2, im_phase)
  # show_result(model, show=5, noShow=10,size=10,imSize=imSize)