## Introduction

Contains two programs, one for training the model (PlantAITrainer) and one for using the model (PlantAIPredictor).

## PIP's

```
pip install tensorflow numpy opencv-python scikit-learn matplotlib
```

## Key Sections of the Program

1. **Loading the Dataset**:
   - `ImageDataGenerator` is used for loading and preprocessing the images, including splitting the dataset into training and validation subsets.
2. **Building the Model**:
   - A basic convolutional neural network (CNN) architecture is used with multiple Conv2D and MaxPooling2D layers followed by a Dense layer for classification.
3. **Training the Model**:
   - The model is trained using the loaded dataset, with accuracy and loss tracked during training and validation.
4. **Plotting the Results**:
   - The training and validation accuracy/loss are visualized using matplotlib.
5. **Saving the Model**:
   - After training, the model is saved as `plant_recognition_model.keras` for future use.

## Credits

ChatGPT 3.5 ;-)
