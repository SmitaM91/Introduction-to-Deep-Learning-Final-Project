# Introduction-to-Deep-Learning-Final-Project

**Facial Keypoint Detection Through CNN With Data Augmentation**

**Project Overview**

For a part of this project, I will create a convolutional neural network (CNN) to perform facial identification by working through the Facial Kepoints Detection Kaggle Competition. I would like to work on AI navigation systems for spaceships someday, therefore the more computer vision-related things I can learn about, the better. To identify key features in faces, I will employ a supervised deep learning model.

Here is the overflow of the project:-

1. Download, Import and Inspect the data
   
2. Clean the data

3. Perform EDA

4. Build an initial CNN, evaluate it against validation

5. Perform hyper parameter optimization to improve results

**Data Importation and Cleaning**

Kaggle's amazing API is a great way to obtain the data for this project. I will manually unzip and place the data for this project in a folder called "data".

**Data Inspection**

I will examine the data's memory requirements using the os.path.size() function, give a summary of the training and test data using the pd.DataFrame.head() method and then explain a cleaning problem in a more understandable way using a hand snippet from the test.csv file. 

The project's data are extremely unorganised. The data are saved in non-rectangular.csv files, however, they encode faces in greyscale. They will require cleaning. Data from greyscale images are often represented as values that represent each pixel's brightness. Unlike RGB images, which call for three dimensions per pixel, there is only one datum per pixel in this case. Unfortunately, the actual image data are encoded as a single string of numbers. For clarification, consider this excerpt from the test dataset:

<img width="194" alt="image" src="https://github.com/user-attachments/assets/c113d3b4-4241-4f5c-bda4-79ce6ff27a04">

To convert the training and test data into a numpy ndarray that can be fed to Keras, we will need to create a custom parser. Because it contains every label we are looking for, the training dataset is slightly more rectangular. Nevertheless, the problem of needing a parser for the picture data persists. As an aside, a user is typically provided with a large number of real photographs while using computer vision. The cv2 library is an excellent resource for transforming images into useful image data. Its imread and resize functions are fantastic tools for turning data into a numpy.ndarray.

Before we rectangularize the datasets, lets first check for null values in the training data. The pd.DataFrame.info() method is an excellent way to see where null values are in a data frame.

All but three of the columns in the training set have null values. Moreover, there are lots of features that only have values for  22717049×100%=32.2%  of their observations. The lack of labels would generally make these observations useless, but removing approximately seventy percent of our dataset is inane. We should instead use an imputation method. The pandas.fillna() method supplies us with a few different ways of imputing labels. We could either forward-fill or back-fill labels from the last/next valid observation using fillna(), which is akin to selecting a random legal value from the feature's probability density function. Another good method is to impute the mean value into each missing value.

It is important to take into account the modelling approaches that will be applied to the data while thinking about imputation strategies. The training data will be sent into a CNN that is connected to an ANN. As the most common (mode) observation in the training set, the worst case situation is that the ANN learns to simply predict the mean value for each imputed label. Therefore, I will use a back-fill imputation rather than a mean value imputation to preserve the diversity of our data. I will forward fill after backfilling to address any potential last-index missing value problems. The fact that the missing data are not properly spaced raises certain concerns because it could result in high counts of duplicated labels. Although we don't think this would happen, we can shuffle the data before imputing to make sure there are no non-independent rows.

**EDA and Augmentation**

I will carry out EDA while I work on data augmentation for my model to ensure that both I and my reader are aware of the true appearance of the photos I am modelling. Let's start by viewing a few faces with keypoints from the training set superimposed on them.

<img width="264" alt="image" src="https://github.com/user-attachments/assets/c3efc68e-ce11-44df-8b48-5b9272658e8e">       <img width="272" alt="image" src="https://github.com/user-attachments/assets/6a135f94-ba9d-481a-a5d2-2cf8ef96e262">



