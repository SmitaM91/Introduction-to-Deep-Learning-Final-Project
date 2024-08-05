# Introduction-to-Deep-Learning-Final-Project

**Facial Keypoint Detection Through CNN With Data Augmentation**

**Project Overview**

For a part of this project, I will create a convolutional neural network (CNN) to perform facial identification by working through the Facial Keypoints Detection Kaggle Competition. I would like to work on AI navigation systems for spaceships someday, therefore the more computer vision-related things I can learn about, the better. To identify key features in faces, I will employ a supervised deep learning model.

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

<img width="264" alt="image" src="https://github.com/user-attachments/assets/c3efc68e-ce11-44df-8b48-5b9272658e8e">       <img width="272" alt="image" src="https://github.com/user-attachments/assets/6a135f94-ba9d-481a-a5d2-2cf8ef96e262">      <img width="274" alt="image" src="https://github.com/user-attachments/assets/ed80271c-78f1-45d4-a18d-de2bbc023a8b">       <img width="273" alt="image" src="https://github.com/user-attachments/assets/78e31171-e3d7-4a04-96df-493d2782fd72">         <img width="272" alt="image" src="https://github.com/user-attachments/assets/3e9e5ae6-ad79-4e18-b9d1-7ae3c98bac56">


Because they are estimated from other data, some of the keypoints are inaccurate. I hope these misses average out to a correct forecast over a large amount of training data and weight adjustments. We discovered in class that feeding a neural network with a variety of potential image orientations is an effective method of training it to generalise the distribution of image data. People pose in a variety of ways for pictures, therefore we should prepare our model for every orientation that might be required. To be clear, we plan to add translations and rotations to our dataset from the original dataset. Of course, we will use the same map to highlight the important spots when the images are translated and rotated.

**Analysis**

Let's have a look at a few of the new upgraded training_images for the purpose of EDA. Each image is now available in five copies, each with a rotation on the set  {−15,0,15}  degrees. These gentle rotations should be representative of what the model could experience in the training set. I will choose five images at random to plot, just like I used to.

<img width="275" alt="image" src="https://github.com/user-attachments/assets/706d1a6f-bcfb-4428-9c97-36350230d802">          <img width="273" alt="image" src="https://github.com/user-attachments/assets/9b53ec01-8002-4cea-af31-3b4e7bb91741">           <img width="270" alt="image" src="https://github.com/user-attachments/assets/d0025eec-fde2-4aaa-bbc9-fc89dc085d5a">          <img width="270" alt="image" src="https://github.com/user-attachments/assets/297f0ca4-3ea7-4b19-9b79-bebefde97c90">               <img width="274" alt="image" src="https://github.com/user-attachments/assets/ca94f14d-8023-4e79-bddd-8d13833b8cf2">

To further diversity the training data, I will now apply linear translations to each image. Although we may perform many other augmentations on the data, I believe that rotation and translation working together are sufficient for this purpose. Reshading, noising up and horizontal flips are some other potential augmentations.

**Model Building and Architecture**

Two parts will make up the model I will employ to predict keypoints: a CNN-style feature extractor and a Dense ANN-style label creator. We discovered in class that iterating through three layers is the normal process for developing a convolutional feature extractor:
[Convolve,Convolve,MaxPool]n 

The next step involves repeating this set of three layers a certain number of times. This architecture selection is a hyperparameter that needs to be maximised. In a convolution layer, a simple mathematical operation is applied to the pixels surrounding a central pixel and the resultant value is recorded as a weight that corresponds to the original pixel. Then, every pixel in the supplied image goes through this procedure once again. In order to "blur" local features for upcoming convolution layers, the MaxPool operation serves as a filter. This helps pick up global features and helps handle chaotic images. Because the succeeding convolution layers will be conducting convolution on already convolved features, which were in turn functions of local features, they will draw from "farther" features relative to the original image as [C, C, MP] iterations grow. Many local traits combine to form a greater global feature. Since we are dealing with facial image data, we want the ANN to "see" the full image before allocating points.

I am going to place BatchNormalization layers after each convolution layer to keep the weights from diverging. An input is mapped by a batch normalisation layer to a Standard Normal variable, which is then used for convolution calculations. It is better to work with smaller, more balanced values while doing addition and multiplication.

For this project, I will experiment with multiple topologies and learning rates as part of my hyperparameter optimisation approach. I will only use the "Adam" optimiser to save training time, although "RMSprop" is also a useful optimiser that can be found in the Keras library. Please feel free to give it a try instead and compare the outcomes.

**Initial Model**

To get started, I will use four [C, C, MP] iterations, each with twice as many filters as the layer before it. LeakyReLU activations are what I will employ because, as our professor pointed out, they allow for greater output diversity for negative valued inputs. It uses a hyperparameter, LeakyReLU  α , which, once I have chosen my architecture, I will experiment with.


**Analysis**

The model functions! It is somewhat bouncy when it comes to validation loss, though, thus we might have to utilise some other callbacks instead of simply a patience option. In order to avoid storing versions of the model that are worse, I will employ a checkpointer. With a basic model up and running, we can begin the process of hyperparameter optimisation. I will verify the values of  η  on the set  {10−5,10−4,10−3,10−2} . It's possible that we could invest a lot of computational effort in optimising  η , yet it doesn't seem worth it given how long a single training cycle takes.

Another thing I will do is maximise the number of [C-C-MP] layers, which can have values between 1 and 5 before the convolution space gets odd (3 X 3) and prevents a 2X2 filter from being properly MaxPooled.

**Results and Analysis**

Let's start by creating some visuals to show how effectively the model worked with the training and validation sets of data. Next, using the test set as a guide, we will create predictions and submit them to Kaggle for evaluation.

<img width="323" alt="image" src="https://github.com/user-attachments/assets/eea93c74-17f5-4891-9887-55eb56c8cddb">

<img width="323" alt="image" src="https://github.com/user-attachments/assets/b876918b-6319-4cda-8eb6-680a471aa211">

**Random Forest**

<img width="264" alt="image" src="https://github.com/user-attachments/assets/d652a657-1eba-4e18-a3b0-52bd6feac0a9">      <img width="256" alt="image" src="https://github.com/user-attachments/assets/9099e9e9-bc63-46dd-a097-ae037e129e39">               <img width="259" alt="image" src="https://github.com/user-attachments/assets/b199eb2b-306c-400d-befa-5560a9a092c4">               <img width="256" alt="image" src="https://github.com/user-attachments/assets/e73fa101-f690-4705-a7d8-9c23f04ac0fc">               <img width="254" alt="image" src="https://github.com/user-attachments/assets/91d00848-438c-4e42-83c7-6a4a87a7c221">

**Analysis**

The random forest appears to consistently produce the same forecasts. This is essentially a null model in which the mean value of each attribute is simply estimated. For facial recognition, this kind of forecast is practically meaningless. For making decisions, the neural network is consequently strongly favoured. While we might attempt to expand the size of the random forest, accuracy is unlikely to converge nearly as quickly as that of the neural network. I would rather improve the neural network through computing time.

**Conclusion**

This project acted as my introduction to facial recognition technology. As a feature extractor, I constructed a convolutional neural network and as a regressor for face important points, I attached it to an artificial neural network. The network performed admirably in recognising eyes and eyebrows, but it struggled to recognise the edges of lips and the points of noses. There is a chance that this error resulted from the various ways people pose for pictures. People will sometimes close their jaws and occasionally smile. When having their photo taken, most people usually have their eyes open.

The neural network performed admirably despite its flaws. As my experiment came to an end, I compared my AI to a random forest regressor. Based on the photos that were presented to it, the random forest completely failed to generate adaptive keypoints. It appears that the null model, or mean value of each feature for each image, is all that the random forest can guess. Future improvements to the nueral network could include longer training times, more enhanced data, and a wider variety of augmentations.

**Bibliography**

Missing Value Imputation - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html





