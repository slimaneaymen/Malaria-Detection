import sys
sys.path.append("libraries/")
from fonction_compteur import *
from fonction_compteur_datagenerator import *
from fonction_compteur_segmentation import *
from fonction_compteur_affiche import *
from skimage.measure import label
from skimage.measure import regionprops
from skimage import img_as_uint
from skimage.io import imread, imsave
from matplotlib import pyplot as plt

image_name = 'G/9994A323901.tif'
image = plt.imread(image_name)
image_mask = decoupe_mask(image, verbose='all')
zoro=np.ma.masked_where(image_mask==0,image_mask)
labeled_list=Hough_by_thres(image, zoro, cells_mean=60, verbose=True,condition=['seuil',1,0.9,1], 
                   edges=None,labeled_list=[], exemple=[False])   
labeled_list=recon_image(labeled_list,verbose='all')
labeled=labeled_list[0] 
#test if region is too small ot too big
classe, amas_cells=test_region(labeled, cells_mean=60, threshold='hough_iter', bord=True)
result=affiche(image, labeled,classe, title=" premier filtrage",boxes=["ellipse","blue"])
save=result[1]
save=np.array(save*255, dtype='uint8')
save=draw_ellipse_perso(save, classe)

save=draw_ellipse_perso(save, amas_cells)
labeled_conv=complet_cells(classe+amas_cells, labeled, verbose='True')
classe=regionprops(labeled_conv)
labeled_conv = img_as_uint(labeled_conv)
imsave('out/'+"im1_labeling.png", labeled_conv)

# def hough_segmentation()