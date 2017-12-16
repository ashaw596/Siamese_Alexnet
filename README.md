Author: Albert Shaw

Implementation of Alexnet-based Siamese network with shared weights. Tested on differentiating images of cat and dog images.
It was trained with shared weights between Siamese networks and with a Contrastive Margin Loss to try to seperate the Cat and Dog images. The final feature descriptor length is 10.
Features are l2-normalized to have a norm of 1.

To run training, first download training images into the trained folder from https://www.kaggle.com/c/dogs-vs-cats/data. Unzip photos directly into that director. Run train_siamese.py for default training run. Results are stored in the records folder.

A result from a training run is included in the records folder. Run visualize_model.py script to visualize the features.

This code is designed to run in python 3 and has not been tested on python 2.

Required packages are in requirements.txt

Results of features with classification test was around 99% accuracy on training data and 75% accuracy on testing data.

Short writeup in Report.docx
