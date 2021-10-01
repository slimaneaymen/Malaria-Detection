# from bigannotator import threshold, image_arithmetic

# add your tests here...

import numpy as np
from bigannotator import napari_experimental_provide_function
import napari
from enum import Enum

# tmp_path is a pytest fixture

class Operation(Enum):
    add = np.add
    subtract = np.subtract
    multiply = np.multiply
    divide = np.divide

def test(data,op):
  threshold, image_arithmetic=napari_experimental_provide_function()     
  Im=image_arithmetic(data,op,data)
  # print(Im)
  return Im
image=np.random.randint(0, 100, (64, 64))
viewer = napari.Viewer()
Im=test(image,Operation.multiply)
# print(Im[0])
viewer.add_image(image,name="Layer 1")
viewer.add_image(Im[0],name="Layer 2")
napari.run() 
