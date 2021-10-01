# napari-bigannotator

[![License](https://img.shields.io/pypi/l/napari-bigannotator.svg?color=green)](https://github.com/sbinnee/napari-bigannotator/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-bigannotator.svg?color=green)](https://pypi.org/project/napari-bigannotator)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-bigannotator.svg?color=green)](https://python.org)
[![tests](https://github.com/sbinnee/napari-bigannotator/workflows/tests/badge.svg)](https://github.com/sbinnee/napari-bigannotator/actions)
[![codecov](https://codecov.io/gh/sbinnee/napari-bigannotator/branch/master/graph/badge.svg)](https://codecov.io/gh/sbinnee/napari-bigannotator)

Cell annotation plugin for Napari viewer

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/docs/plugins/index.html
-->

## BigAnnotator : A napari plug-in to annotate cells and nuclei in bio-medical images, powered by machine learning
BigAnnotator is (will be) a plug-in for Napari viewer. It will help users to
annotate cells and nuclei in bio-medical images, taking advantage of napari's
capability to view large n-dim images interactively and python's availability to
use powerful machine learning algorithms from open-science library, such as
`sklearn`, `opencv`, `scikit-image`, or even deep learning libraries such as
`TensorFlow` and `PyTorch`. 

Napari is a promising image viewer that can handle large n-dimensional images
interactively for python. Python has been extremely successful in open-science
by replacing many proprietary tools, for an example MatLab, and thus bringing
open contribution and fast development to open source algorithms and tools. 

Napari itself is just an image viewer, though it provides interactive data
manipulation using python, which may not be an ideal for those who do not know
the language. The way Napari sees to build an open-science ecosystem around it
is via plug-ins. 


## Features
Its main feature is the interactive active learning. ML and DL show remarkable
accuracy for segmentation task in papers. However, these models would not work
very well on your own data, simply because they are not optimized for your data.
As a result, their predictions are not going to be perfect and almost always
need human efforts in the end. BigAnnotator will provide ML algorithms to help
users to correct annotation. Then BigAnnotator will use the corrected annotation
to fine-tune the model to be optimized for your own data.

- Contour assistance

    ML algorithms will guide the contour for you

- Correct the predicted contour

    Napari's drawing tool will let you correct the contour

- Active learing

    Continuously update ML/DL models to fit to your data and the corrections you
    made


## How it works
BigAnnotator uses multiple ML and DL models. There are **SegModel**,
**ContModel**, and **ClsModel**. 

- **SegModel** is a DL model. It will perform segmentation task on the
displayed area. It is never or not expected to be updated frequently. 

- **ContModel** is a ML, DL model or mixture of both. It will help users to find
  contour for each ROI (region of interest). 

- **ClsModel** is a ML model. It will classify each instance and assign labels
  to them, if necessary. It is going to be updated frequently, thus it needs to
  be a light model. 

1. Initialize SegModel and perform the initial segmentation.
2. User corrects contours and labels. ContModel will help to correct user
   modification.
3. Initialize ClsModel and perform classification.
4. Go back to 1 and repeat. 
5. Once a while update SegModel with modified annotation. 


## Active learning
Here, active learning means two things; continuously updating ML/DL model, and
taking into account user modification. To be Interactive, deep learning may not
be preferable for the moment because it is slow to train deep neural networks
and even so if CUDA GPU is not available. Ensemble models would be the one. ex)
random forest, XGBoost, etc...


## Formats
Zarr is the default file format. It supports multi-scale images, which is an
essential feature for large image data. It is also a recommended file format by
napari.

SVG is probably the annotation file format. It is designed to be scale
invariant, which makes perfect candidate for multi-scale annotation along with
zarr. Note that SVG is meant for 2-dimensional graphics.


## Other tools
### [AnnotatorJ](https://github.com/spreka/annotatorj)
AnnotatorJ is a Fiji/ImageJ plugin for annotation. It uses a deep learning model
to provide contour assistance to segment cells and nuclei. It works sometimes
but it also breaks in many cases. AnnotatorJ heavily inspired contour assistance
functionality of BigAnnotator. 

### [Ilastik](https://www.ilastik.org/)
Ilastik is a popular tool, especially in bio-medical image community, for
segmentation task. It provides a set of machine learning algorithms through a
graphical user interface, so that users can easily perform segmentation task. It
also provides active learning feature, which means that the model will take into
account user modification and becomes a better suited model for users' data.
Ilastik is not designed for large data. Recently it started supporting DL model
too. If you do not have large image data to annotate, consider using Ilastik
instead of BigAnnotator.

### [ImJoy](https://imjoy.io/)
ImJoy is designed to advocate easy use of deep learning in your web browser. But
ImJoy is not capable of handling large n-dimensional image data and modifying
annotation is not the strongest feature.

## Installation

Clone this repository and install it via [pip]:

    git clone https://gitlab.com/seongbin.lim/bigannotator.git

    cd bigannotator/

    pip install -e .

<!--
You can install `napari-bigannotator` via [pip]:

    pip install napari-bigannotator
-->

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-bigannotator" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://gitlab.com/seongbin.lim/bigannotator/-/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

## Contact
Any new ideas are welcome
- Seongbin Lim : seongbin.lim@polytechnique.edu
