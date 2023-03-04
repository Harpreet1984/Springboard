 #                       PlantDisease Dataset Disease Recognition  

![This is an image](https://github.com/Harpreet1984/Springboard/blob/main/Capstone3/Images/Screen%20Shot%202023-03-04%20at%2012.12.16%20AM.png)


###                                   Abstract
						
The first step to good agricultural yield is protecting the plants from diseases. Early plant disease recognition and prevention is the first step in this regard. But manual disease recognition is time-consuming and costly. This is one of those use cases, where deep learning can be used proactively for great benefit. Using deep learning, we can recognize plant diseases very effectively. Large scale plant disease recognition using deep learning can cut costs to a good extent.


The goal of this project is to develop algorithms that can accurately diagnose a disease based on an image.We will use deep learning for disease recognition on the PlantVillage dataset using deep learning and PyTorch.



### Table of Contents

1. [ Introduction. ](#desc)

	1.1 [Fundamentals](#subparagraph1)
2. [ Methodology. ](#desc)

	2.1 [Importing the dataset](#subparagraph1)
	
	2.2 [Exploring the dataset](#subparagraph1)	
	
	2.3 [Data Preparation for training](#subparagraph1)
	
	2.4 [Building the Model Architecture](#subparagraph1)
	
	2.5 [Model Training](#subparagraph1)
	
	2.6 [Results](#subparagraph1)
3. [ Further Improvements of study. ](#usage)
4. [ Conclusion](#usage)
5. [ References](#usage)


<a name="desc"></a>
## 1. Introduction

Plant diseases, a deterioration in the original condition of the plant that upsets and changes its vital functions, are common problems that farmers and agronomists face from season to season. Various aspects can cause plant diseases, such as environmental conditions, the presence of pathogens, and the variety of crops located nearby the plant. In the initial period of most of the diseases, symptoms can be observed by looking at alterations of the physical properties of plants, such as alteration in color, size, and shape, due to the disease.

Plant diseases significantly disturb the progress of the plant and are recognized as one of the factors which affects food security. Consequently, fast, and accurate identification is essential to contain the spread of disease. However, the identification procedure is hampered by the complexity of the procedure. It is found that experienced agronomists and plant biologists find difficulty in differentiating certain plant diseases because of the multitude of symptoms that occur. The, where misdiagnosis leads to inadequate or inadequate treatment . An automation system at this level will support and reduce the workload of farmers and agronomists .

Over the last few years, the exponential growth in computing power, particularly in the graphics processing unit (GPU), has resulted in machine learning algorithms being able to run efficiently on servers due to their ability to compute processes in parallel. Additionally, the introduction of GPU platforms such as Compute Unified Device Architecture (CUDA) from Nvidia has introduced a new platform for developers to accelerate applications which are compute intensive. This is accomplished by utilising the computing power of GPU in a way that is optimised for computation of complex algorithms such as gradient descent and back-propagation and gradient descent . Advances and breakthroughs in technology have enabled deep learning approaches to be widely used in many applications in nearly every field .

The deep learning model is created and based on combination of neural network architectures. In contrast to the conventional neural network architecture, which comprises of a few hidden layers only, a deep learning model may contain hundreds of hidden layers. This is one of the reasons which make deep learning, generally, can have higher accuracy in comparison with conventional methods. The Convolutional Neural Network (CNN) , which is characterized by its function for feature extraction in image recognition and image classification , is considered as one of the deep learning models. With different types of design, CNN model can conduct diverse types of operations on images.

The main contribution of the current study is the development of a deep learning model in the PyTorch environment which can identify plant diseases of various plant species based on plant leaf images on the edge.

### 1.1 Fundamentals <a name="subparagraph1"></a>
Plant diseases are generally classified into the following 3 categories: -
1) Viral- Viruses are very small infectious RNA parasite that often damages or kills the plant’s infected cells. 
2) Bacterial- Bacteria is a member of group of microscopic single celled organisms which can only be seen through a microscope. Bacteria often overwhelm the immune system and results in severe and harmful diseases in living organisms.
 3) Fungal- Fungi are organisms that lack chlorophyll and thus they do not have the ability to photosynthesize their own food.Fungi are especially harmful during preharvest and postharvest of crops. They produce highly toxic, hallucinogenic chemicals that have affected millions of animals including humans and still continue to do so.
 
 ![This is an image](https://github.com/Harpreet1984/Springboard/blob/main/Capstone3/Images/fig1.jpg)

<a name="usage"></a>
## 2. Methodology

We have approached the given problem by using  deep learning framework PyTorch. For our end goal we decided to develop a model using  deep learningl and transfer learning technique. 

This analysis was conducted using Python through Jupyter notebook. In-built libraries and methods were used to run the machine learning models. When needed, functions were defined to simplify specific analyses or visualizations.


### 2.1 Importing the dataset <a name="subparagraph1"></a>
This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.


 ![This is an image](https://github.com/Harpreet1984/Springboard/blob/main/Capstone3/Images/plantdisease.webp)
                         

_Fig 2 : Plant Diseases_                              
<a name="usage"></a>
### 2.2 Exploring the dataset <a name="subparagraph1"></a>
The dataset contains 38 classes of crop disease pairs

We have 14 unique varieties of plants.

We have 26 types of images that show a particular disease in a particular plant.

There are 70295 images for training
The below figure shows number of images in each disease category


![This is an image](https://github.com/Harpreet1984/Springboard/blob/main/Capstone3/Images/fig3.jpg)

_Fig 3 : Number of images in each category_ 
<a name="usage"></a>
### 2.3 Data Preparation for training <a name="subparagraph1"></a>
The TorchVision datasets subpackage is a convenient utility for accessing well-known public image and video datasets. You can use these tools to start training new computer vision models very quickly.We used subclass torchvision.datasets.ImageFolder which helps in loading the image data when the data is arranged in a specific way

Normalising image input - Data normalization is an important step which ensures that each input parameter (pixel, in this case) has a similar data distribution. This makes convergence faster while training the network. Data normalization is done by subtracting the mean from each pixel and then dividing the result by the standard deviation. The distribution of such data would resemble a Gaussian curve centered at zero. For image inputs we need the pixel numbers to be positive, so we might choose to scale the normalized data in the range [0,1] or [0, 255]. For our data-set example, we need to transform the pixel values of each image (0-255) to 0-1 as neural networks. The entire array of pixel values is converted to torch tensor and then divided by 255.

DataLoader is a subclass which comes from torch.utils.data. It helps in loading large and memory consuming datasets. It takes in batch_size(we are taking 32) which denotes the number of samples contained in each generated batch.Setting shuffle=True shuffles the dataset. It is heplful so that batches between epochs do not look alike. Doing so will eventually make our model more robust.

![This is an image](https://github.com/Harpreet1984/Springboard/blob/main/Capstone3/Images/Fig%204%20.png)

_Fig 4 : Batchsize 32 images for dataloader__

<a name="usage"></a>

### 2.4 Building the model architecture  <a name="subparagraph1"></a>
ResNet short for ‘Residual Networks’ is a neural network. It can have a very deep network and it is a subclass of convolutional neural networks. It does this by understanding the residual representation functions rather than learning and understanding the signal representation right away. The new concept introduced in ResNet was shortcut connections or skip connections, to fit the preceding layer input to next following layer without changing it. This shortcut attachment allows it to have an in-depth network. We had to pick ResNet as our model.	

 Below are the benefits of using ResNet: 

a) Problems with Plain Network: Usually Conventional deep learning networks contain convolutional layers interconnected with fully connected layers for the classification job, without skipping or changing any connection. Due to deeper layers, the complication of vanishing or exploding gradients may appear in the plain network.	

![This is an image](https://github.com/Harpreet1984/Springboard/blob/main/Capstone3/Fig.%204.png)						
						
_Fig 5 Vanishing Gradient Problem; Image source: Medium.com_ 
					
						


As seen from the above figure the deeper networks suffer more from vanishing/exploding gradient problem than shallow networks. 

b) Skipping Connection in the ResNet: To solve the complication in the areas of vanishing/ and exploding gradients, a skipping interconnection is joined so that the raw input x to the next layer is the output given by the previous layer after few weight layers.	

![This is an image](https://github.com/Harpreet1984/Springboard/blob/main/Capstone3/Fig6.png)

The skip connection in the diagram above is labeled “identity.” It allows the interconnection of networks to learn the identity function, which facilitate it to pass the input to the block needed without passing or sending it through the other weight layers.Hence, the output is given as: H(x) = F(x) + x. The weight layers are there to understand the types of residual mapping like F(x) = H(x)−x. This allows us to stack additional layers and build a deeper
network, offsetting the vanishing gradient by allowing the network interconnections to skip between some layers which it feels are less needed for training. Even if vanishing gradient occur in the weight layers, we will still have the feature x to get back to the preceding layers. 


c) ResNet vs Plain Networks: When a plain network is used, a low layer network is always better. For eg. It is better to use plain network on an 18-layer network than a 34-layer network. For a high layer network Resnet performs better because in a deep network it beats plain networks by introducing skip connections, hence eliminating vanishing gradient problem.

![This is an image](https://github.com/Harpreet1984/Springboard/blob/main/Capstone3/Fig%207%20.png)

*Fig 7 Plain Networks v ResNet; Image source:Medium.com*

If we compare18-layer plain network and18- layer ResNet, the difference isn’t much. This is because vanishing gradient problem does not appear for shallow networks. However, when ResNet is used on 34-layer network, it performs way better. Here vanishing gradient problem has been solved by using skip connections. 

<a name="usage"></a>
### 2.5 Model Training <a name="subparagraph1"></a>
Here the previously clean and transformed data is trained on the training set. The images in ‘Train data’ folder will be used for training our neural network, while the ’Validate data’ will be used to validate results obtained from our trained model. During training the model will analyze the input data set and find its own meaning. Later on the ‘Test data’ will be used to test our model. Our trained neural network will be put to test on a set of images and we will know if the model works as expected or if there are any flaws in it.

Before we trained the model,we defined a utility function an evaluate function, which will perform the validation phase, and a fit_one_cycle function which will perform the entire training process. In fit_one_cycle, we have use some techniques:

**Learning Rate Scheduling** : Instead of using a fixed learning rate, we will use a learning rate scheduler, which will change the learning rate after every batch of training. There are many strategies for varying the learning rate during training, and the one we’ll use is called the “One Cycle Learning Rate Policy”, which involves starting with a low learning rate, gradually increasing it batch-by-batch to a high learning rate for about 30% of epochs, then gradually decreasing it to a very low value for the remaining epochs.

**Weight Decay** :  We also use weight decay, which is a regularization technique which prevents the weights from becoming too large by adding an additional term to the loss function.

**Gradient Clipping** : Apart from the layer weights and outputs, it also helpful to limit the values of gradients to a small range to prevent undesirable changes in parameters due to large gradient values. This simple yet effective technique is called gradient clipping.

<a name="usage"></a>
### 2.6 Results <a name="subparagraph1"></a>
The results of our classifier gives us an accuracy of 96.2 %  when trained on 10 epochs.
The results highly depend on the number of epochs the model is trained on and also on the amount of testing dataset.We have achieve this accuracy by pre-processing the images to make the model more generic, split the data set into a number of batches and finally build and train the model.

|             | Learning rate | Training loss  | Validation loss | Validation accuracy|
| ----------- | -----------   |-------------   |-----------------|--------------------|
| Epoch 0     |   0.00812     |	0.7511	       | 0.8085	         | 0.7636             |
| Epoch 1     |   0.00000     |	0.1235	       | 0.0263          | 0.9925             |


![This is an image](https://github.com/Harpreet1984/Springboard/blob/main/Capstone3/Images/fig%209.jpg)
![This is an image](https://github.com/Harpreet1984/Springboard/blob/main/Capstone3/Images/fig%2010.jpg)

<a name="subparagraph1"></a>

## 3. Further Improvements of Study

We can develop web application which gives the project a simple and clean look where a user just has to select a plant leaf image and the application will display the disease with the highest predicted percentage.

We are currently constrained to the classification of single leaves, facing up, on a homogeneous background. While these are straightforward conditions, a real world application should be able to classify images of a disease as it presents itself directly on the plant. Indeed, many diseases don’t present themselves on the upper side of leaves only (or at all), but on many dierent parts of the plant. Thus, new image collection eorts should try to obtain images from many dierent perspectives, and ideally from settings that are as realistic as possible.

We evaluated the applicability of Resnet  for the said classification problem. We can focus on other  popular architectures like AlexNet and GoogLeNet which were designed in the context of the "Large Scale Visual Recognition Challenge" for the Image Classifications.

## 4. Conclusion

Humans for centuries have evaluated and produced plantbased food products for fiber, medicine, home, etc. Diseases in plants are just one of the many hazards that must be considered while cultivating crops. Thus, it is important that we enhance the food quality and look to stable agricultural sector as it ensures a nation of food security. The project “Plant Disease Detection using Deep Learning” is aimed at building a neural network capable of detecting crop species and  common diseases. We were able to build a ResNet9 model using convolutional neural network that can recognize images with an accuracy of 99.23% using Pytorch. 

## 5. References

1]  S.P. Mohanty, D.P. Hughes, M. Salathé
Using deep learning for image-based plant disease detection Front. Plant Sci., 7 (2016), p. 1419, 10.3389/fpls.2016.01419

2] A. Fuentes, S. Yoon, S.C. Kim, D.S. Park
 A robust deep-learning-based detector for real-time tomato plant diseases and pests recognition
Sensors, 17 (9) (2017), p. 2022, 10.3390/s17092022

3] YOSUKE TODA AND FUMIO OKURA
 How Convolutional Neural Networks Diagnose Plant Disease
https://spj.science.org/doi/10.34133/2019/9237136

4] Yan Guo,1,2Jin Zhang,3Chengxin Yin,4Xiaonan Hu,1
Plant Disease Identification Based on Deep Learning Algorithm in Smart Farming
https://www.hindawi.com/journals/ddns/2020/2479172/#data-availability

