# Malaria-Detection
This is a project to Detect malaria. The interactive graphical user interface (Napari plugin) developed allows you to: Reconstruct high resolution images from low resolution ones, and to retrieve phase information using Fourier Ptychography Microscopy,  segment red blood cells using (classic,advanced) algorithms, classify cells into healthy/infected, visualize the results, and allows for individual control of blood cells at the various imaging, processing and classification steps. 


## Setup

1- You should first create a virtual envirement (in my case I used conda):
   `conda create -n yourenvname python=3.8.10 anaconda`
  Then activate the envirement:
  `conda activate yourenvname`

2- Install Napari:

  `https://napari.org/tutorials/fundamentals/installation`
  The editable mode was used (as Napari is in an alpha stage, it may be better to stay in touch to new changes) :
  ```
  git clone https://github.com/napari/napari.git
  cd napari
  pip install -e .[all]
  ```
3- Try to launch napari to check if it works, by running in your terminal: 
  `napari`

  A new window should appear showing the Napari default window, if not please refer to the Napari website cited above.

4- Then you should install some packages:
  Stardist: `pip install stardist`
  Cellpose: `pip install cellpose`
  Tensorflow: `pip install tensorflow`
  opencv: `pip install opencv-python`
  LightPipes: `pip install LightPipes`
  scikit-image: `pip install -U scikit-image`


5- Then you should set the path in Terminal to : 
  `Malaria Detection/Malaria-Detection/cell_annotator/`, where the file `setup.py` is located, then run : `pip install .`

6- After that, run in terminal:  `napari` (Napari window should be closed before running this) 

7- In the menu bar (of Napari window) go to `plugins --> napari-cell_annotator`, where you should get the list of plugins (Image_preprocessing, segmentation, Image classification, ...). To open a plugin click on its name in the list.

8- In case you didn't find `napari-cell_annotator` in `plugins --> napari-cell_annotator`, you can even:
  * Solution 1: 
    Go to `plugins --> install/uninstall plugins..` then check if in `installed plugins` list you have `napari-cell_annotator` (where       the check box next to `napari-cell_annotator` should be checked, if not check it and close this window then go to step 7.
    
  * Solution 2: 
    Close Napari window, then set the path in your terminal to where `_dock_widget.py` is located, then run : 
    `pytest _dock_widget.py` this will allow you to visualize the errors you have ( which are generally related to a package that you       should install). If so, install the needed packages then go to step 5.     

For a better undrestanding of how to use the different plugins, please refer to the file : `Malaria_detection_Plugin_Documentation`.
For more explanation about the work (technical aspects), it can be found on the papers folder.



