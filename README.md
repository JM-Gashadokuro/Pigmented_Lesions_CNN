# Pigmented_Lesions_CNN
## Description
This is a simple app consisting of two programs - one generating and training a model of a Convolutional Neural Network meant to classify pigmented skin lesions from 
photos into 7 categories:  
Actinic keratoses, Basal cell carcinoma, Benign keratosis-like lesions, Dermatofibroma, Melanocytic nevi, Melanoma and Vascular lesions.  
The other is meant to provide a simple Graphical User Interface, to make for an easier, more user-friendly way of querying the model for predictions. To get a prediction, simply click the "Browse"button on top of the window to browse through your files and select a folder containing photos you wish to classify. Once that's done, you'll see a list of all the .jpgphotos contained in that folder in the left panel of the window. Simply click a photo on the list to preview the photo on the right panel, and get a prediction below the photo.  
  
![Desktop 23-01-2023 15-45-47-538](https://user-images.githubusercontent.com/48767765/214070503-b5d690bd-f730-40f2-8973-faeaef1feae5.png)
## Used technologies
The first part of the app (the one used for generating and training a model) mostly relies on Keras to build and train the model. It also relies on pandas for csv import
as well as converting a certain variable to categorical type, numpy for calculating standard deviation and mean, as well as converting lists into arrays.  
Other than that, there's pillow for loading and resizing images and sklearn (requiring also scikit-learn) for randomly splitting data into train/validate/test subsets.
