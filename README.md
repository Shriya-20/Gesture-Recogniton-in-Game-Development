# Gesture Recognition with 3DCNN and ConvLSTM

## Objective:
The goal is to recognize human hand gestures in real-time using a custom-built model trained on the [Jester Dataset](https://www.qualcomm.com/developer/software/jester-dataset). This project aims to detect gestures like "swiping left","swiping right", "thumb up", "thumb down","zooming in" and "zooming out" to simulate keypress events during gameplay or media interaction.

## Dataset Overview:
- **Dataset**: Jester (148,092 labeled video clips)
- **Classes**: 27 distinct hand gestures (e.g., swiping, zooming, thumbs up)
- **Training/Validation/Test split**: 
  - Training: 118,562 clips
  - Validation: 14,787 clips
  - Test: 14,743 clips
- **Format**: JPG images, 100px height, 12 FPS, stored in 1 GB TGZ archives.

## Preprocessing:
1. **Annotations**: Load, filter, and preprocess using DataFrames.
2. **Data Augmentation**: (not implemented yet).
3. **Standardizing frame count**: Ensures consistent frames per gesture.(36)
4. **Data Generators**: Custom batch generation with frame standardization and one-hot encoding of labels.

## Model Architectures:
1. **3DCNN-4 layers (SGD)**:  
   - Input shape: (36, 32, 32, 3)
   - Optimizer: SGD
   - Accuracy: 81.7%
   - Issues: Struggled with "Zooming In/Out" and "Thumb Up".
   
2. **3DCNN-2 ConvLSTM (Adam)**:  
   - Optimizer: Adam
   - Accuracy: 85.21%
   - Improved performance and class separation.
   
3. **3DCNN-3 ConvLSTM (RMSprop)**:  
   - Accuracy: 83.95%
   - Performance stable, slight drop in overall results.

4. **3DCNN-4 layers + ConvLSTM-1 layer (Adam)**:  
   - Best performance: 87.85%
   - Most stable and accurate for gestures.

## Mixed Precision:
Uses a combination of 16-bit and 32-bit floating-point numbers for improved performance, optimized for NVIDIA GPUs with Tensor Cores.

## Results:
- **Swiping Right**: High precision and recall.
- **Swiping Left**: High precision and recall.
- **Thumb down**: good results.
- **Thumb Up**: Lower precision, often misclassified.
- **Zooming Gestures**: Struggles a little with differentiation.


- The project generates an executable `.exe` for easy deployment. 
- The application recognizes gestures in real-time and triggers keypress events (e.g., swiping left for the left arrow key).
- Works alongside media players or games, enabling easy interaction through gestures.

## Usage:
- Run the `.exe` file and interact with your gestures.
- The top 3 gesture predictions will be displayed in real-time.

## References:
- [Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
- https://openaccess.thecvf.com/content_ICCVW_2019/papers/HANDS/Materzynska_The_Jester_Dataset_A_Large-Scale_Video_Dataset_of_Human_Gestures_ICCVW_2019_paper.pdf

