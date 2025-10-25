# Artificial-Intelligence-in-Ophthalmology-Fundus-Based-Deep-Learning-for-Glaucoma-Detection

## Project Metadata
### Authors
- **Team:** Zeina Moammer Alabido, g202423020
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** KFUPM

## Introduction
Glaucoma is one of the well-known Retinal diseases that happen due to factors that 
lead to irreversible sight loss. It occurs when the fluids of the eye do not drain correctly, 
consequently, that leads to eyeball pressure. There is an optic nerve connected in 
between their eye and the brain, which will be affected by this pressure. This kind of 
disease is not treated and tackled, the vessels within the eye tissue would be burst. 
Approximately 3.6 million people might be affected with blindness because of 
glaucoma. In addition, research estimated that about 76 million of humans they think 
they might have glaucoma, another 94.4 million they are more expected to be 
affected by glaucoma in 2023, and what goes the problem more seriously is that 50% 
of the people they have glaucoma they do not know, because they ignore the 
symptoms they might feel it is normal[2]. 
There are different types of glaucoma: 
1- Primary open-angle Glaucoma: In this type, the channels that work on fluids 
draining are blocked. It might send fewer warnings about having glaucoma, 
but it is better if it has been detected early [2]. 
2- Acute angle-closure Glaucoma: this type causes symptoms that need urgent 
attention, such as a medicine to lower the eye pressure sometimes used before 
the surgery [2]. 
3- Secondary Glaucoma: this type of Glaucoma might be caused because of an 
injury, taking certain medicine, or an illness, for example, steroids [2]. 
Artificial and Machine learning can have an important favor in the early detection of 
Glaucoma, whether in its early ages, or even later. With AI, once we can detect the 
Glaucoma, the best and appropriate treatment will be given to the patient who has 
those diseases. 

## Problem Statement
The problem of Glaucoma is a significant medical health problem that should be 
addressed carefully. Nowadays, the Artificial Intelligence world has become 
demandable in the field of medicine, since it assists the doctors in the early discovery 
of a serious medical problem. However, the data becomes an obstacle in this field, 
such as it is not available publicly, it is limited to specific region, etc. Moreover, patients 
sometimes might not accept the idea of having their image has been used for 
medical analysis to be used for research purposes, so the result is, small, and simple 
dataset. Also, deep learning models require high dimensional data, because the DL 
models are too complex, and it introduces non-linearity, so it requires more complex 
datasets. 

## Application Area and Project Domain
The dataset has been used in this study it has been collected from Kim's Eye Hospital, 
Yeongdeungpo-gu, Seoul, South Korea. Unfortunately, the data set availability is 
extremely limited, despite some of them are publicly available to be used, but very 
few ones. 

## What is the paper trying to do, and what are you planning to do?
To develop a model that able to classify the Glaucoma Retinal Diseases using 
Fundus Images and leverage the beauty of Multiclassification. In addition, Generalize the results of the model to other datasets (if available). Moreover, If the model successes, we should generalize that the model will work well on other retinal disease classifications. 


# THE FOLLOWING IS SUPPOSED TO BE DONE LATER

### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/report.pdf)

### Reference Paper
- [A foundation model for generalizable disease detection from retinal images](https://www.nature.com/articles/s41586-023-06555-x)

### Reference Dataset
- [Machine learn for Glaucoma Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/1YRRAC)

## Project Technicalities

### Terminologies
- **Convolutional Neural Networks** Convolutional Neural Networks (CNNs) is an enhanced version of Artificial Neural Networks (ANNs), it has an automatic feature extraction technique from grid matrix dataset, such as images. CNNs are used mostly in Computer Vision Applications. CNNs are consist of input layer, pooling layer, and fully connected dense layer, ended with an output layer to give the predicted label. What makes CNNs challenging is that it is requires a lot of resources for training the model, which is computationally expensive, and it is prone to overfit if the dataset is not big enough, which requires large amount of data.
- **MobileNetV2 Architecture:** MobileNetV2 is a pretrained model, and a powerful CNN architecture has been established for light-weight employment such as mobile devices, or embedded vision applications. What makes MobileNetV2 interesting is that it introduces the concept of inverted residual blocks, and bottlenecks, which results in high accuracy. Also, it uses ReLU6 [10], which is an activation function that clips the output at 6, preventing numerical instability and making the model more suitable. 

### Problem Statements
- **Problem 1:** Dataset Dependency: Most of the papers found have been using private datasets, which are not publicly available.
- **Problem 2:** It is limited to a specific scope. In other words, most of the papers have collected datasets from their own region. Consequently, the models generalize to all types of images in different countries.
- **Problem 3:** Most of the paper focuses on classification, rather than trying more advanced techniques such as segmentation, object detection, or even image captioning.
- **Problem 4:** Universal Model: Do these models work to classify or detect other retinal diseases? This is one of the limitations in the previous papers.

### Loopholes or Research Areas
- **Simple Deep Learning Models:** The topic of Glaucoma classification focused on classifying into Glaucoma or Non-Glaucoma, which means applying only binary classification.
- **High Quality Computational Resources:** Training requires significant GPU compute resources and strong machines, since the neural networks require a complex and large datasets to train and test the models, as a result, strong GPUs must be used.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** -Hypothesis: This research will build a strong and robust deep learning model to classify glaucoma cases.
3. **Enhanced Data Augmentation:** Apply the needed steps for data preparation, processing, and augmentation to prepare the data for model training and testing.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced multi-classification glaucoma detection using tensorflow.keras library. The solution includes:

- **Customized-CNN Model:** The proposed CNN used for this project is as follows: Conv2D + ReLU: it consists of 32 filters, and the size of each filter is 3×3, over the input image of size 224×224×3. The output of this layer is going through ReLU activation function to introduce non-linearity and scale the data. Next, followed by Batch Normalization, which normalize the activations within each batch, by calculating the meaning and the variance of data, and then normalize. Batch normalization followed by Max Pooling2D, which works on reducing the dimensionality by reducing the spatial size of feature map. In this case, the Max Pooling applied with the size (2,2). The next layer is another Conv2D + ReLU with 64 filters, each 3×3, which will start learning more hidden patterns, and more complex features, followed by Batch Normalization and Max Pooling2D. Next is Conv2D + ReLU with 128 filters of size 3×3, start to go deep inside the network and reach higher level of learning, followed by Batch Normalization, and Max Pooling2D. These 3 Conv2D layers followed by Flattening layer, which converts the 3D feature map to 1D vector, preparing them for the next layer, which is the Dense Layer. The Dense Layer consists of 256 perceptron with ReLU activation, to learn non-linear combinations. After that, Dropout with 0.4 has been applied to avoid overfitting, followed by Batch Normalization. The Last layer, which gives the prediction of multi-classification using SoftMax.
- **Custom MobileNetV2 Architecture:** This is a CNN model from TensorFlow Kera Applications. Basically, the weights of “ImageNet” dataset have been loaded, the top fully connected layers of MobileNetV2 have been freeze, because we will build our own. We have used the feature extractor of mobileNetV2 with the custom layers that have been added. The base of MobileNetV2 is followed by a batch normalization, with a global average pooling2D. This is followed by a dense layer that has a 128 + ReLU, with a dropout equals to 0.3. The last layer is a dense layer to produce the output using SoftMax function. 
- **ReLU Activation Function** Using ReLU Activation function to introduce non-linearity between layers of architecture, for more efficient performance.
- **Adam Optimizer:** is an optimization technique used to adjust the learning rate during the training, it combines between the advantages of Root Mean Square Propagation (RMSProp), and the momentum.
- **Categorical Cross-Entropy Loss Function:** categorical cross entropy loss function is used to calculate the loss by the models they are classifying, and it is used when we have more than two classes to classify (multi-classification problems). As an evaluation Metrex for the quality of the models, Categorical Cross Entrop Loss [18]has been used to assess how neural networks performs good on data.

### Key Components
- **`model.py`**: Contains the modified UNet architecture and other model components.
- **`train.py`**: Script to handle the training process with configurable parameters.
- **`utils.py`**: Utility functions for data processing, augmentation, and metric evaluations.
- **`inference.py`**: Script for generating images using the trained model.

## Model Workflow
The workflow of the Enhanced Stable Diffusion model is designed to translate textual descriptions into high-quality artistic images through a multi-step diffusion process:

1. **Input:**
   - **Text Prompt:** The model takes a text prompt (e.g., "A surreal landscape with mountains and rivers") as the primary input.
   - **Tokenization:** The text prompt is tokenized and processed through a text encoder (such as a CLIP model) to obtain meaningful embeddings.
   - **Latent Noise:** A random latent noise vector is generated to initialize the diffusion process, which is then conditioned on the text embeddings.

2. **Diffusion Process:**
   - **Iterative Refinement:** The conditioned latent vector is fed into a modified UNet architecture. The model iteratively refines this vector by reversing a diffusion process, gradually reducing noise while preserving the text-conditioned features.
   - **Intermediate States:** At each step, intermediate latent representations are produced that increasingly capture the structure and details dictated by the text prompt.

3. **Output:**
   - **Decoding:** The final refined latent representation is passed through a decoder (often part of a Variational Autoencoder setup) to generate the final image.
   - **Generated Image:** The output is a synthesized image that visually represents the input text prompt, complete with artistic style and detail.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/enhanced-stable-diffusion.git
    cd enhanced-stable-diffusion
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

4. **Generate Images:**
    Once training is complete, use the inference script to generate images.
    ```bash
    python inference.py --checkpoint path/to/checkpoint.pt --input "A surreal landscape with mountains and rivers"
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to bla, bla, bla for the amazing team effort, invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to ABC-organization for providing the computational resources necessary for this project.
