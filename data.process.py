from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directories where the training and validation data are stored
train_dir = 'datasets/food-101/images/train'
val_dir = 'datasets/food-101/images/test'

# Data augmentation and normalization for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Normalization for validation data
val_datagen = ImageDataGenerator(rescale=1./255)

# Loading images from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize images
    batch_size=32,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
