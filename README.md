# deepfake-detector

This project implements a DeepFake detection pipeline using **MTCNN for face extraction** and **ResNet18** for classification. It is designed to work on both **real and fake videos**, processing them into face crops, training a CNN classifier, and evaluating performance using metrics like **AUC-ROC** and **accuracy**.

**File Structure**

extract_faces.py - MTCNN-based face extraction from video frames
run_extraction.py - One-time script to extract face crops for the dataset
resnet_classifier.py - ResNet18 architecture with 2-class output
evaluate.py - Evaluation functions with AUC, accuracy, ROC curve
main.py - Main training & evaluation pipeline
requirements.txt - All dependencies for this project

**Model Info**-

--Backbone: ResNet18 (ImageNet pretrained)

--Face Detector: MTCNN from facenet-pytorch

--Loss Function: CrossEntropy

--Optimizer: Adam

**Notes**

--Face extraction only needs to be run once unless you change the dataset.

--Make sure you're using GPU for faster training. (Colab, Kaggle, or local CUDA)

--Supports adding video-level aggregation and temporal modeling in future updates.

To setup & run:
**Clone the repo**
```bash
git clone https://github.com/yourusername/deepfake-detector.git
cd deepfake-detector
```

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Mount your data (if using Google Drive or similar)**
Ensure your video dataset is structured as:
```
data/train/real
data/train/fake

data/test/real
data/test/fake
```

**Extract Face Crops (run once)**
```bash
python run_extraction.py
```

**Train and Evaluate the model**
```bash
python main.py
```

*Make The Necessary adjustments to the code as per your requirements*
