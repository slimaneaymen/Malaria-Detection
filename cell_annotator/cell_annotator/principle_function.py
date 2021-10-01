# next, add the location of the file to the path:
import sys

from matplotlib import cm
sys.path.append('C:/Users/slimane/Desktop/Design Project/slimane_lio/libraries')
import os
import time
import tensorflow as tf
sys.path.append(os.getcwd()) #folder with code
import pandas as pd
import cv2
from skimage.io import imread, imsave
from fonction_compteur_segmentation import Hough_by_thres, recon_image, test_region, detect_para
from fonction_compteur_datagenerator import decoupe_mask, sauvegarde_imagette
from fonction_compteur_affiche import affiche,draw_ellipse_perso
from fonction_compteur import ouvrir, complet_cells
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters    import  rank
from skimage.morphology import disk,black_tophat
from skimage.measure    import regionprops
from skimage.io         import imsave

travel_in="C:/Users/slimane/Desktop/Design Project/slimane_lio/G" # travel picture input 
travel_out="C:/Users/slimane/Desktop/Design Project/slimane_lio/out/save_cell_images/" # travel output image 



def extract_cells(image, labeled,raw, coords_para, coords_distrac,cells_mean=60, size1=71, size2=71, 
                     travel_output=os.getcwd()+"_output/", tophat=False, zoro=False,mask=False):
    debut=time.time()
    #existence of folder
    if not os.path.exists(travel_output):
      os.makedirs(travel_output)
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
    for region in regionprops(labeled):
        p = p+1 
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
                
              #imagette image
              # image_sortie[coords[:,0]-region.bbox[0]+begin_x, coords[:,1]-region.bbox[1]+begin_y]=\
              # image[coords[:,0],coords[:,1]]
              if (mask!=True):
                image_sortie[center1-int(size1/2):center1+int(size1/2), center2-int(size2/2):center2+int(size2/2)]=\
                image[dx0-diffx1:dx1+diffx2,dy0-diffy1:dy1+diffy2]
              else:
                image_sortie[center1-int(size1/2):center1+int(size1/2), center2-int(size2/2):center2+int(size2/2)]=\
                image[dx0-diffx1:dx1+diffx2,dy0-diffy1:dy1+diffy2]
                imsave(travel_output+str(p)+'.png', image_sortie[center1-int(size1/2):center1+int(size1/2), center2-int(size2/2):center2+int(size2/2)])
                image_sortie_mask[coords[:,0]-region.bbox[0]+diffx1, coords[:,1]-region.bbox[1]+diffy1]=\
                labeled[coords[:,0],coords[:,1]]
                a = image_sortie_mask[coords[:,0]-region.bbox[0]+diffx1, coords[:,1]-region.bbox[1]+diffy1]
                print(image_sortie_mask.shape,image_sortie_mask.dtype,image_sortie_mask)
                plt.imshow(image_sortie_mask,cmap='gray')
                plt.imshow(image_sortie,cmap='gray')
                imsave(travel_output+'mask_'+ str(p)+'.png', image_sortie_mask)
              #imagette tophat
              if tophat:
                  image_tophat[coords[:,0]-region.bbox[0]+begin_x, coords[:,1]-region.bbox[1]+begin_y]=\
              black_para[coords[:,0],coords[:,1]]
              
#               title=str(region.label)+'_'+str(int(round(xc)))+'x'+str(int(round(yc)))+'.png'
              #travel_output_new = travel_output + str(region.label) + "/"
              if (mask!=True):
                title=str(raw)+'.png'
              else:
                title='mask'+'.png'
#               if coords_para is False:
#                   pass
#               else:
#                   if len(list_para)>0:
#                       if len(list_para[0])==0:
#                           pass
#                       else:
#                           for i in range(len(list_para)-1,-1,-1):
#                               if list_para[i][0] in coords[:,0]:
#                                   indice=np.where(coords[:,0]==list_para[i][0])[0]
#                                   resultat=np.where(coords[indice,1]==list_para[i][1])[0]
#                                   if len(resultat)!=0:
#                                       place=indice[resultat][0]
#                                       if len(resultat)==1:               
#                               #        print(i,"est dans la matrice 2 en",place, "mat2["+str(place)+",:]=",matrice2[place])
#                                           list_para.pop(i)
#                                           infected=True
#                                       else :
#                                           print("problem a coords parasite is more than 1 time in a cells")
#                                           print("coords = ",place)
# #                        print(valeur,"en matrice 2 à",place)
#               if infected:
#                   #pas de distracteur
#                   pass
#               else:
#                   if coords_distrac is False:
#                       pass
#                   else:
#                       for i in range(len(list_distrac)-1,-1,-1):
#                           if list_distrac[i][0] in coords[:,0]:
#                               indice=np.where(coords[:,0]==list_distrac[i][0])[0]
#                               resultat=np.where(coords[indice,1]==list_distrac[i][1])[0]
#                               if len(resultat)!=0:
#                                   place=indice[resultat][0]
#                                   if len(resultat)==1:               
#                           #        print(i,"est dans la matrice 2 en",place, "mat2["+str(place)+",:]=",matrice2[place])
#                                       list_distrac.pop(i)
#                                       distrac=True
#                                   else :
#                                       print("problem a coords parasite is more than 1 time in a cells")
#                                       print("coords = ",place)
#   #                        print(valeur,"en matrice 2 à",place)
              
              
#               if infected:
#                   travel_output_new = travel_output + 'infected_' + str(region.label) + "/" 
#                   #title='infected_'+title
#               else:
#                   if distrac:
#                     travel_output_new = travel_output + 'distrac_' + str(region.label) + "/" 
#                     #title = 'distrac_'+title
#                   else:
#                     travel_output_new = travel_output + 'healthy_' + str(region.label) + "/" 
#                     #title='healthy_'+title
#     #    plt.close("all")
#     #    plt.figure()
#     #    plt.imshow(image_sortie)
#               if not os.path.exists(travel_output_new):
#                 os.makedirs(travel_output_new)
#               imsave(travel_output_new+title, image_sortie)
#               if tophat:
#                   imsave(travel_tophat+title, image_tophat, check_contrast=False)
#     fin=time.time()
#     #print(fin-debut)
# #    print(coords_para)
#     if not coords_para is False:
#         if len(list_para)>0:
# #            print("il reste des parasites non detecter")
#             print("longueur de liste para =",len(list_para))
#             return list_para
#     else:
#         return None


def global_segmentation(image,label=True, mask=True,travel_out = os.getcwd(),path1 = os.getcwd(),
        path2 = os.getcwd(),name1 = 'csv', name2 ='labels',size1=84,size2=84, save=True,verbose='all',
        cells_mean=60,segment='otsu canny',close=True,boxes=['ellipse','blue'],
        temps=True,hough='circular cv2 skimage'):
    #extraction images and choice of image. 

    #extraction image in folder and create list of travel.
    # images=[]
    # file=os.listdir(travel_in)
    # print('travel_in',travel_in)
    # names =[]

    # for i in file:
    #     images.append(travel_in+'/'+i)
    #     names.append(i)
    # print(images)
    # #choice your image.
    # image=images[5]
    # name = names[5].split(".")[0]
    # print(image)
    # print(len(images))
    print('image.dtype',image.dtype)
    raw_select = 0
    #if is a raw image choose the first
    # image=ouvrir(image[0]) 
    # print(travel_in)
    # print('titles',titles)
    raw_imgs = image
    # a,b = titles.split(' ')
    # titles = b
    # print('a',a,'b',b,'titles',titles)
    #extract parasite coords.
    if label :
        # coords_para = pd.read_csv("C:/Users/slimane/Desktop/Design Project/slimane_lio/csv/"+ name +".csv", sep=';')
        coords_para = pd.read_csv(path1 +'/' +name1 +".csv", sep=';')
        #coords_para=detect_para (image,zoro, verbose=True, title='image', thres=10)
    else :
        coords_para=[[],[]]

    coords_para = np.array(coords_para)



    # labeled = imread('C:/Users/slimane/Desktop/Design Project/slimane_lio/labeled/'+titles+' labeling.png')
    labeled = imread(path2+'/'+name2+ '.png')
    classe = regionprops (labeled)
    
    print('nb cells detected = ',len(classe))
    print('len(raw_imgs)',len(raw_imgs))
    # for i in range(0,len(raw_imgs[0])):
    #     if len(raw_imgs)==0:        
    #         image = raw_imgs[0]
    #     else:
    #         image = raw_imgs[0][i,:,:]    
        #normalise image
    image = raw_imgs
    print(image.dtype,image.dtype)
    if image.dtype!='uint8':
        img_cr=(image -image.min())/(image.max()-image.min())
        image=img_cr*255
        del(img_cr)
        image=np.array(image, dtype='uint8')
    travel_output_modi = travel_out + '_' + 'image'+'/'
    extract_imagette(image, labeled, coords_para=coords_para, coords_distrac=False,raw=0,cells_mean=60, size1=size1, size2=size1,  travel_output=travel_output_modi,mask=mask)
    print('image.shape',image.shape)
