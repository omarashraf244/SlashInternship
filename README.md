# SlashInternship 

### Step 1: Extracting Images from a Zip File
- The code begins by importing the necessary libraries, notably `zipfile` for extracting files from a zip archive.
- It extracts the contents of a zip file located at `zip_file_path` using `ZipFile.extractall()` method.

### Step 2: Importing Libraries and Modules
- Several libraries and modules are imported:
    - `os` for operating system functionalities.
    - `cv2` for image processing using OpenCV.
    - `numpy` for numerical computations.
    - `train_test_split` and `LabelEncoder` from `sklearn.model_selection` and `sklearn.preprocessing`, respectively, for splitting data into training and validation sets and encoding labels.
    - `Sequential`, `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense` from `tensorflow.keras.layers` for defining the CNN architecture.
    - `ImageDataGenerator`, `EarlyStopping`, `ModelCheckpoint`, and `LearningRateScheduler` from `tensorflow.keras.preprocessing.image` and `tensorflow.keras.callbacks` for data augmentation and callbacks during training.

### Step 3: Loading Images
- A function `load_images_from_dir()` is defined to load images from directories.
- It iterates through each file in the directory, reads the image using OpenCV (`cv2.imread()`), converts color space from BGR to RGB, and appends images and corresponding labels to lists.
- The function returns a list of images and their labels.

### Step 4: Data Loading
- Paths to directories containing different categories of images (bags, clothing, accessories) are specified.
- Images and labels are loaded for each category using the `load_images_from_dir()` function.

### Step 5: Data Preprocessing
- Images are resized to (150, 150) using `cv2.resize()` for consistency in size.
- Pixel values are normalized to the range [0, 1] by dividing by 255.0.

### Step 6: Label Encoding
- Labels are encoded using `LabelEncoder()` from scikit-learn to convert categorical labels into numerical format.

### Step 7: Train-Validation-Test Split
- The dataset is split into training, validation, and test sets using `train_test_split()` function.
- Training set is further divided into training and validation sets.

### Step 8: Model Building
- A CNN model is defined using `Sequential()` and various layers such as convolutional (`Conv2D`), pooling (`MaxPooling2D`), and fully connected (`Dense`) layers.
- ReLU activation function is used for hidden layers, and softmax activation is used for the output layer since it's a multi-class classification problem.
- The model is compiled with `adam` optimizer and `sparse_categorical_crossentropy` loss function.

### Step 9: Model Training
- The model is trained using `fit()` method on the training data.
- Training is performed for 10 epochs initially.

### Step 10: Model Testing
- Model performance is evaluated on the test set using `evaluate()` method.

### Step 11: Data Augmentation
- Data augmentation is applied using `ImageDataGenerator()` to increase the diversity of training data by performing random transformations such as rotation, shifting, shearing, zooming, and flipping.
- This helps prevent overfitting and improves generalization.

### Step 12: Callbacks
- Callbacks like early stopping and model checkpointing are defined to monitor the validation loss and save the best model weights during training.

### Step 13: Model Training with Callbacks
- The augmented training data is fed to the model for further training.
- Callbacks are applied during training.

### Step 14: Model Evaluation (After Data Augmentation)
- The model is evaluated on the test set again after data augmentation.

### Step 15: Plotting Training and Validation Curves
- Training and validation loss as well as accuracy curves are plotted using `matplotlib.pyplot`.
