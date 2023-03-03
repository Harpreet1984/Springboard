 #                       PlantVillage Dataset Disease Recognition  

![This is an image](https://github.com/Harpreet1984/Springboard/blob/main/Capstone3/PlantVillage-Dataset-Disease-Recognition-using-PyTorch-e1669809682570.png)


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
3. [ Limitations of study. ](#usage)

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
 
 ![This is an image](https://github.com/Harpreet1984/Springboard/blob/main/Capstone3/Screen%20Shot%202023-03-02%20at%206.07.13%20PM.png)

<a name="usage"></a>
## 2. Methodology

We have approached the given problem by using  deep learning framework PyTorch. For our end goal we decided to develop a model using  deep learningl and transfer learning technique. 

This analysis was conducted using Python through Jupyter notebook. In-built libraries and methods were used to run the machine learning models. When needed, functions were defined to simplify specific analyses or visualizations.


### 2.1 Importing the dataset <a name="subparagraph1"></a>
This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.

<a name="usage"></a>
### 2.2 Exploring the dataset <a name="subparagraph1"></a>
The dataset contains 38 classes of crop disease pairs

We have 14 unique varieties of plants.

We have 26 types of images that show a particular disease in a particular plant.

There are 70295 images for training
The below figure shows number of images in each disease category


![This is an image](https://github.com/Harpreet1984/Springboard/blob/main/Capstone3/Screen%20Shot%202023-03-03%20at%201.58.10%20AM.png)

<a name="usage"></a>
### 2.3 Data Preparation for training <a name="subparagraph1"></a>
The TorchVision datasets subpackage is a convenient utility for accessing well-known public image and video datasets. You can use these tools to start training new computer vision models very quickly.We used subclass torchvision.datasets.ImageFolder which helps in loading the image data when the data is arranged in a specific way

Normalising image input - Data normalization is an important step which ensures that each input parameter (pixel, in this case) has a similar data distribution. This makes convergence faster while training the network. Data normalization is done by subtracting the mean from each pixel and then dividing the result by the standard deviation. The distribution of such data would resemble a Gaussian curve centered at zero. For image inputs we need the pixel numbers to be positive, so we might choose to scale the normalized data in the range [0,1] or [0, 255]. For our data-set example, we need to transform the pixel values of each image (0-255) to 0-1 as neural networks. The entire array of pixel values is converted to torch tensor and then divided by 255.


<a name="usage"></a>
