from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Load and preprocess the dataset
def load_dataset(data_dir, img_size=(150, 150), batch_size=32):
    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)  # Normalize images

    # Load training data
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    # Load validation data
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, val_generator


# 2. Build the CNN model
def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# 3. Train the model
def train_model(model, train_generator, val_generator, epochs=10):
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
    )
    return history


# 4. Plot training results
def plot_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


# 5. Main function to execute
if __name__ == "__main__":
    # Path to your dataset directory
    dataset_dir = "C:\\Users\\evert\\PycharmProjects\\PlantAI\\PlantAITrainer\\dataset"

    # Print the categorical labels (handy for later assigning class indices, see the PlantAIPredictor project)
    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    print(train_generator.class_indices)

    # Load the data
    train_gen, val_gen = load_dataset(dataset_dir)

    # Get input shape from data generator and number of classes
    input_shape = train_gen.image_shape
    num_classes = train_gen.num_classes

    # Build and compile the model
    model = create_model(input_shape, num_classes)

    # Train the model
    history = train_model(model, train_gen, val_gen, epochs=10)

    # Plot training history
    plot_results(history)

    # Save the trained model
    model.save('plant_recognition_model.keras')

