
# Face Detection


## Introduction
Face Recognition is one of the major challenges that researchers in Machine Learning - Deep Learning have been facing and dealing with. This problem can be applied in various fields, especially in areas requiring high accuracy and security such as eKYC in E-Commerce and identity recognition through surveillance cameras (CCTV). We will divide this problem into 2 main issues: Face Detection and Face Verification. To delve deeper into understanding and applying a complete Face Recognition task, we will explore and infer two famous networks for these two issues: MTCNN (Multi-task Cascaded Convolutional Networks) and FaceNet. 
### Dataset
The dataset consists of 10 directories, with each directory containing images of a person named after the directory. You can create additional directories with the New_Student.ipynb file.
Some images included in the dataset:

![](Face_Detection/asset/img/1.png)

Two Google links are used to manage students during the software execution:

Attendance with check marks: [Link](https://docs.google.com/spreadsheets/d/1y9ppEx41uNmy4MkcUhrsR5vMCNWFoBDWn8crXTAPoe8/edit#gid=0)

Attendance with timestamps: [Link](https://docs.google.com/spreadsheets/d/1OIUp_4M6lHowcaEhAd8LcKfQDsiUvqoV3y5vPZr-a3g/edit#gid=0)
## Overview of MTCNN (Multi-task Cascaded Convolutional Networks):

MTCNN is a network designed to detect faces in images or frames within videos, with three distinct network layers representing three main stages: P-Net, R-Net, and O-Net (PRO!).

### Stage 1: P-Net
Initially, an image typically contains more than one face, and faces often vary in size. We need a method to identify all these faces at different sizes. MTCNN provides a solution by using image resizing to create a series of copies from the original image with different sizes, forming an Image Pyramid.

![](Face_Detection/asset/img/img_pyramid.png)

For each resized copy of the original image, a 12x12 pixel kernel with a stride of 2 traverses the entire image to detect faces. Because the copies of the original image have different sizes, the network can easily recognize faces of different sizes, even though it uses only one kernel with a fixed size (larger image, larger face; smaller image, smaller face). Then, the kernels cropped from above are passed through the P-Net (Proposal Network). The network's output is a series of bounding boxes within each kernel, with each bounding box containing four corner coordinates to determine its position within the kernel (normalized to the range (0,1)) and the corresponding confidence score.

![](Face_Detection/asset/img/P_net.png)

To eliminate redundant bounding boxes across images and kernels, we use two main methods: setting a confidence threshold to remove boxes with low confidence scores and using NMS (Non-Maximum Suppression) to remove boxes with overlapping ratios (Intersection Over Union) above a certain threshold. The image below illustrates the NMS process, where redundant boxes are removed, leaving only one box with the highest confidence score.

![](Face_Detection/asset/img/orig_n_img.png)

After removing unnecessary boxes, we convert the coordinates of the boxes back to the coordinates of the original image. Since the coordinates of the box have been normalized to the range (0,1) corresponding to the kernel, the task now is to compute the length and width of the kernel based on the original image, then multiply the normalized coordinates of the box by the size of the kernel and add them to the corresponding corner coordinates of the kernel. The result of this process will be the coordinates of the corresponding box on the original-sized image. Finally, we resize the boxes back to square shapes, take the new coordinates of the boxes, and feed them into the next network, the R-Net.

### Stage 2: R-Net
![](Face_Detection/asset/img/R_Net.png)

The R network (Refine Network) performs steps similar to the P network. However, it also utilizes a method called padding to insert zero-pixels into the missing parts of the bounding box if the bounding box exceeds the image boundary. All bounding boxes are resized to 24x24, considered as one kernel, and fed into the R network. The output results are also new coordinates of the remaining boxes, which are then passed to the next network, the O network.

### Stage 3: O-Net
![](Face_Detection/asset/img/O_Net.png)

Finally, the O network (Output Network) performs similarly to the R network, resizing the boxes to 48x48. However, the output results of the network are not just the coordinates of the boxes anymore. Instead, it returns three values: 4 coordinates of the bounding box (out[0]), 5 landmark points on the face, including 2 eyes, 1 nose, and 2 sides of the mouth (out[1]), and the confidence score of each box (out[2]). All these results are stored in a dictionary with 3 keys as mentioned above.

![](Face_Detection/asset/img/face_detection.png)

After completing the Face Detection part with MTCNN, faces can be extracted from the images. Next, for the Face Verification task, we will use the FaceNet network to distinguish and cluster the faces.
## FaceNet (A Unified Embedding for Face Recognition and Clustering):

The next section is about Face Verification. The main task of this problem is to evaluate whether the current face image matches the information, the face of another person already present in the system or not. To approach this problem, first, we need to understand some concepts in the FaceNet Paper.

### Basic Concepts:
#### Embedding Vector: 
It's a vector with a fixed dimension (usually smaller than regular Feature Vectors), learned during the training process, and represents a set of features responsible for classifying objects in the transformed space. Embeddings are very useful in finding Nearest Neighbors in a given Cluster, based on the distance-relationship between embeddings.
#### Inception V1:
A CNN architecture introduced by Google in 2014, featuring Inception blocks. These blocks allow the network to learn in parallel structures, meaning that one input can be fed into multiple different Convolution layers to produce different results, which are then concatenated into one output. This parallel learning enables the network to capture more details, extract more features compared to traditional CNNs. Additionally, the network applies 1x1 Convolution blocks to reduce the network's size, making the training process faster.

![alt text](Project_Face_Detection/img/inceptionv1.png)

#### Triple Loss:
Instead of using traditional loss functions, where we only compare the network's output values with the ground truth of the data, Triplet Loss introduces a new formula consisting of 3 input values: anchor xia: the output image of the network, positive xip: the image of the same person as the anchor, and negative xin: the image of a different person from the anchor. α is the margin (additional margin) between positive and negative pairs, the necessary minimum deviation between two value ranges, fxia is the embedding of xia. The formula indicates that the desired distance between two embeddings, fxia and fxip, must be less than the value compared to the pair fxia and fxin. Our goal is to make the difference between the two sides of the formula as large as possible, or in other words, ‖fxia- fxip ‖22 must be minimum and ‖fxia- fxin ‖22 must be maximum. To make the network "harder" to learn (or in other words, learn more), the selected positive point must be as far away as possible from the anchor, and the selected negative point must be as close as possible to the anchor, to make the network "encounter" the worst cases. The general Loss function presented in the paper is the following formula:
iN‖fxia- fxip ‖22-  ‖fxia- fxin ‖22+ α+

Gif image of the learning process using the Triplet Loss function:
![](Face_Detection/asset/gif/learning.gif)

### Implementation Process:
#### Training Procedure:
- Use a Dataset with a large number of different individuals, each with a certain number of images.
- Build a DNN network used as a Feature Extractor for the Dataset, resulting in a 128-Dimensional embedding. In the paper, there are two representative networks: Zeiler&Fergus and InceptionV1.
- Train the DNN network to produce embeddings capable of good recognition, including using l2 normalization (Euclidean distance) for the output embeddings and optimizing the network parameters using Triplet Loss.
- The Triplet Loss function will use the Triplet Selection method, selecting embeddings for the best learning process.

![](Face_Detection/asset/img/architecture.png)

#### Inference Procedure:
- Feed the face image to be classified into the Feature Extractor network, obtaining an embedding.
- Use l2 function and compare it with other embeddings in the existing embedding set. The classification process will be similar to the k-NN algorithm with k = 1.## Get API
### Step 1: Access: [Link](https://developers.google.com/sheets/api/quickstart/python?hl=vi)

![](Face_Detection/asset/img/step_1.png)

### Step 2: Find the Prerequisites section and click on the link [A Google Cloud project](https://developers.google.com/workspace/guides/create-project)

### Step 3: After successfully accessing, click on Go to Create a Project

![](Face_Detection/asset/img/step_3.png)

### Step 4: Enter the necessary information (It is recommended to use a school email for easy registration) and select Create:

![](Face_Detection/asset/img/step_4.png)

### Step 5: After creation, select APIs and services → Library, find Google Drive API and Google Sheet API and enable them (click Enable)

![](Face_Detection/asset/img/step_5.png)

![](Face_Detection/asset/img/step_5_2.png)

### Step 6: Then select Credentials → CREATE CREDENTIALS → Service account, fill in the necessary information and copy the email address:

![](Face_Detection/asset/img/step_6.png)

### Step 7: Choose the role as Owner and click DONE

![](Face_Detection/asset/img/step_7.png)

### Step 8: Click on the email

![](Face_Detection/asset/img/step_8.png)

### Step 9: Select KEYS → ADD KEY → Create new key → JSON → CREATE

### Step 10: Link the downloaded file into the code:

![](Face_Detection/asset/img/step_10.png)

Share Google Sheet:

![](Face_Detection/asset/img/step_10_2.png)
![](Face_Detection/asset/img/step_10_3.png)

Run face_rec.py (Recognize multiple faces) or face_rec_cam.py (Recognize a single face)

After recognition, the software will mark the attendance of students:

![](Face_Detection/asset/img/demo.png)
![](Face_Detection/asset/img/tracking.png)
![](Face_Detection/asset/img/track_timing.png)
## Run
To execute the commands in the terminal:

Run the following command to tightly crop the faces:

```bash
python Face_Detection/src/align_dataset_mtcnn.py  Face_Detection/dataset/FaceData/raw Face_Detection/dataset/FaceData/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25
```

Run the following command to train the model:

```bash
python Face_Detection/src/classifier.py TRAIN Face_Detection/dataset/FaceData/processed Face_Detection/models/20180402-114759.pb Face_Detection/models/facemodel.pkl --batch_size 1000
```

Run the following command to activate the camera:

```bash
python Face_Detection/src/face_rec_cam.py 
```

or 

```bash
python Face_Detection/src/face_rec.py 
```

Make sure to navigate to the correct directory before executing these commands.