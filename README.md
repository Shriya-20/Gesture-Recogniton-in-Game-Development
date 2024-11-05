# Gesture-Recogniton-in-Game-Development
## **Gesture Recognition with 3DCNN and LSTM**

## Objective
This project aims to recognize human hand gestures in real-time using a custom-built model trained on the [Jester Dataset](https://www.qualcomm.com/developer/software/jester-dataset). The system aims to detect gestures such as "No Gesture", "swiping left," "swiping right," "thumb up," "thumb down," "zooming in," and "zooming out" to simulate keypress events during gameplay or media interaction.

## Dataset Overview
- **Dataset**: Jester
  - **Total Clips**: 148,092 labeled video clips
  - **Classes**: 27 distinct hand gestures (e.g., swiping, zooming, thumbs up)
  - **Training/Validation/Test Split**:
    - Training: 118,562 clips
    - Validation: 14,787 clips
    - Test: 14,743 clips
  - **Format**: JPG images, 100px height, 12 FPS, stored in 1 GB TGZ archives

## Preprocessing
1. **Annotations**: Load, filter, and preprocess using DataFrames.
2. **Standardizing Frame Count**: Ensures consistent frames per gesture (36 frames).
3. **Data Generators**: Custom batch generation with frame standardization and one-hot encoding of labels.

## Model Architectures
| Model                            | Optimizer       | Accuracy | Notes |
|----------------------------------|-----------------|----------|-------|
| 3DCNN-4 layers (elu, SGD)        | SGD (0.001)    | 81.7%    | Challenges with Zoom In/Out |
| 3DCNN-2 ConvLSTM-2 (elu, Adam)   | Adam (0.0001)  | 85.2%    | Better class separation |
| 3DCNN-3 layers + ConvLSTM-3      | RMSprop (0.0001)| 83.95%  | No significant gain |
| 3DCNN-4 layers + ConvLSTM-1      | Adam (0.0001)  | 87.85%   | Least misclassification |
| 3DCNN-4 layers + ConvLSTM-2      | Adadelta (1.0) | 90.55%   | Best accuracy and class separation |

Some points about the architecture are mentioned here. Refer notebook for more details about architecture.

The models were evaluated on validation data. 


## Mixed Precision
This project utilizes a combination of 16-bit and 32-bit floating-point numbers for improved performance, optimized for NVIDIA GPUs with Tensor Cores.

## Results
- **Swiping Right**: High precision and recall.
- **Swiping Left**: High precision and recall.
- **Thumb Down**: Good results.
- **Thumb Up**: Lower precision, often misclassified.
- **Zooming Gestures**: Struggles with differentiation.

- The application recognizes gestures in real-time and triggers keypress events (e.g., swiping left corresponds to the left arrow key).
- Compatible with media players and games, enabling intuitive interaction through gestures.

## Usage
1. Run the `.exe` file.
2. Interact with your gestures.
3. The top 3 gesture predictions will be displayed in real-time.

## Environment Setup
### Dependencies
- Python 3.8
 **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   
### Instructions to Run the Code
1. **Clone this repository**:
   ```bash
   git clone https://github.com/Shriya-20/Gesture-Recogniton-in-Game-Development.git
   cd Gesture-Recognition-in-Game-Development
  You can use my pre-trained model in this repository or train your own before running the application.
## Instructions to Run the Code
2. **Run the application**:
   ```bash
    python app.py
   
## Usage:
- The top 3 gesture predictions will be displayed in real-time in real time, gestures simulate keypress events enabling you to customize it based on required functionality.

## References:
- [Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
- https://openaccess.thecvf.com/content_ICCVW_2019/papers/HANDS/Materzynska_The_Jester_Dataset_A_Large-Scale_Video_Dataset_of_Human_Gestures_ICCVW_2019_paper.pdf


   
