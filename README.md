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
- **Diffusion Model:** A generative model that progressively transforms random noise into coherent data.
- **Latent Space:** A compressed, abstract representation of data where complex features are captured.
- **UNet Architecture:** A neural network with an encoder-decoder structure featuring skip connections for better feature preservation.
- **Text Encoder:** A model that converts text into numerical embeddings for downstream tasks.
- **Perceptual Loss:** A loss function that measures high-level differences between images, emphasizing perceptual similarity.
- **Tokenization:** The process of breaking down text into smaller units (tokens) for processing.
- **Noise Vector:** A randomly generated vector used to initialize the diffusion process in generative models.
- **Decoder:** A network component that transforms latent representations back into image space.
- **Iterative Refinement:** The process of gradually improving the quality of generated data through multiple steps.
- **Conditional Generation:** The process where outputs are generated based on auxiliary inputs, such as textual descriptions.

### Problem Statements
- **Problem 1:** Dataset Limitation: Most of the papers found have been using private datasets, which are not publicly available.
- **Problem 2:** It is limited to a specific scope. In other words, most of the papers have collected datasets from their own region. Consequently, the models generalize to all types of images in different countries.
- **Problem 3:** Most of the paper focuses on classification, rather than trying more advanced techniques such as segmentation, object detection, or even image captioning.
- **Problem 4:** Universal Model: Do these models work to classify or detect other retinal diseases? This is one of the limitations in the previous papers.

### Loopholes or Research Areas
- **Evaluation Metrics:** Lack of robust metrics to effectively assess the quality of generated images.
- **Output Consistency:** Inconsistencies in output quality when scaling the model to higher resolutions.
- **Computational Resources:** Training requires significant GPU compute resources, which may not be readily accessible.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign the model architecture to improve efficiency and balance image quality with faster inference.
2. **Advanced Loss Functions:** Integrate novel loss functions (e.g., perceptual loss) to better capture artistic nuances and structural details.
3. **Enhanced Data Augmentation:** Implement sophisticated data augmentation strategies to improve the model’s robustness and reduce overfitting.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced stable diffusion model using PyTorch. The solution includes:

- **Modified UNet Architecture:** Incorporates residual connections and efficient convolutional blocks.
- **Novel Loss Functions:** Combines Mean Squared Error (MSE) with perceptual loss to enhance feature learning.
- **Optimized Training Loop:** Reduces computational overhead while maintaining performance.

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
