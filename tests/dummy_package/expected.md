<span style="float:right;">[[source]](www.dummy.com/my_project/tests/dummy_package/dummy_module.py#L1)</span>

### Dense class


```python
tests.dummy_package.dummy_module.Dense(
    units,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
```


Just your regular densely-connected NN layer.

`Dense` implements the operation:
`output = activation(dot(input, kernel) + bias)`
where `activation` is the element-wise activation function
passed as the `activation` argument, `kernel` is a weights matrix
created by the layer, and `bias` is a bias vector created by the layer
(only applicable if `use_bias` is `True`).

Note: if the input to the layer has a rank greater than 2, then
it is flattened prior to the initial dot product with `kernel`.

__Example__

```python
# as first layer in a sequential model:
model = Sequential()
model.add(Dense(32, input_shape=(16,)))
# now the model will take as input arrays of shape (*, 16)
# and output arrays of shape (*, 32)

# after the first layer, you don't need to specify
# the size of the input anymore:
model.add(Dense(32))
```

__Arguments__

- __units__: Positive integer, dimensionality of the output space.
- __activation__: Activation function to use
    (see [activations](../activations.md)).
    If you don't specify anything, no activation is applied
    (ie. "linear" activation: `a(x) = x`).
- __use_bias__: Boolean, whether the layer uses a bias vector.
- __kernel_initializer__: Initializer for the `kernel` weights matrix
    (see [initializers](../initializers.md)).
- __bias_initializer__: Initializer for the bias vector
    (see [initializers](../initializers.md)).
- __kernel_regularizer__: Regularizer function applied to
    the `kernel` weights matrix
    (see [regularizer](../regularizers.md)).
- __bias_regularizer__: Regularizer function applied to the bias vector
    (see [regularizer](../regularizers.md)).
- __activity_regularizer__: Regularizer function applied to
    the output of the layer (its "activation").
    (see [regularizer](../regularizers.md)).
- __kernel_constraint__: Constraint function applied to
    the `kernel` weights matrix
    (see [constraints](../constraints.md)).
- __bias_constraint__: Constraint function applied to the bias vector
    (see [constraints](../constraints.md)).

__Input shape__

nD tensor with shape: `(batch_size, ..., input_dim)`.
The most common situation would be
a 2D input with shape `(batch_size, input_dim)`.

__Output shape__

nD tensor with shape: `(batch_size, ..., units)`.
For instance, for a 2D input with shape `(batch_size, input_dim)`,
the output would have shape `(batch_size, units)`.


----

<span style="float:right;">[[source]](www.dummy.com/my_project/tests/dummy_package/dummy_module.py#L113)</span>

### ImageDataGenerator class


```python
tests.dummy_package.dummy_module.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format="channels_last",
    validation_split=0.0,
    interpolation_order=1,
    dtype="float32",
)
```


Generate batches of tensor image data with real-time data augmentation.

The data will be looped over (in batches).

__Arguments__

- __featurewise_center__ `bool`: Boolean.
    Set input mean to 0 over the dataset, feature-wise.
- __samplewise_center__: Boolean. Set each sample mean to 0.
- __featurewise_std_normalization__: Boolean.
    Divide inputs by std of the dataset, feature-wise.
- __samplewise_std_normalization__: Boolean. Divide each input by its std.
- __zca_whitening__: Boolean. Apply ZCA whitening.
- __zca_epsilon__: epsilon for ZCA whitening. Default is 1e-6.
- __rotation_range__: Int. Degree range for random rotations.
- __width_shift_range__: Float, 1-D array-like or int
    - float: fraction of total width, if < 1, or pixels if >= 1.
    - 1-D array-like: random elements from the array.
    - int: integer number of pixels from interval
        `(-width_shift_range, +width_shift_range)`
    - With `width_shift_range=2` possible values
        are integers `[-1, 0, +1]`,
        same as with `width_shift_range=[-1, 0, +1]`,
        while with `width_shift_range=1.0` possible values are floats
        in the interval `[-1.0, +1.0)`.
- __height_shift_range__: Float, 1-D array-like or int
    - float: fraction of total height, if < 1, or pixels if >= 1.
    - 1-D array-like: random elements from the array.
    - int: integer number of pixels from interval
        `(-height_shift_range, +height_shift_range)`
    - With `height_shift_range=2` possible values
        are integers `[-1, 0, +1]`,
        same as with `height_shift_range=[-1, 0, +1]`,
        while with `height_shift_range=1.0` possible values are floats
        in the interval `[-1.0, +1.0)`.
- __brightness_range__: Tuple or list of two floats. Range for picking
    a brightness shift value from.
- __shear_range__: Float. Shear Intensity
    (Shear angle in counter-clockwise direction in degrees)
- __zoom_range__: Float or [lower, upper]. Range for random zoom.
    If a float, `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
- __channel_shift_range__: Float. Range for random channel shifts.
- __fill_mode__: One of {"constant", "nearest", "reflect" or "wrap"}.
    Default is 'nearest'.
    Points outside the boundaries of the input are filled
    according to the given mode:
    - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
    - 'nearest':  aaaaaaaa|abcd|dddddddd
    - 'reflect':  abcddcba|abcd|dcbaabcd
    - 'wrap':  abcdabcd|abcd|abcdabcd
- __cval__: Float or Int.
    Value used for points outside the boundaries
    when `fill_mode = "constant"`.
- __horizontal_flip__: Boolean. Randomly flip inputs horizontally.
- __vertical_flip__: Boolean. Randomly flip inputs vertically.
- __rescale__: rescaling factor. Defaults to None.
    If None or 0, no rescaling is applied,
    otherwise we multiply the data by the value provided
    (after applying all other transformations).
- __preprocessing_function__: function that will be applied on each input.
    The function will run after the image is resized and augmented.
    The function should take one argument:
    one image (NumPy tensor with rank 3),
    and should output a NumPy tensor with the same shape.
- __data_format__: Image data format,
    either "channels_first" or "channels_last".
    "channels_last" mode means that the images should have shape
    `(samples, height, width, channels)`,
    "channels_first" mode means that the images should have shape
    `(samples, channels, height, width)`.
    It defaults to the `image_data_format` value found in your
    Keras config file at `~/.keras/keras.json`.
    If you never set it, then it will be "channels_last".
- __validation_split__: Float. Fraction of images reserved for validation
    (strictly between 0 and 1).
- __interpolation_order__: int, order to use for
    the spline interpolation. Higher is slower.
- __dtype__: Dtype to use for the generated arrays.

__Examples__

Example of using `.flow(x, y)`:
```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)
# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)
# here's a more "manual" example
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
```
Example of using `.flow_from_directory(directory)`:
```python
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
```
Example of transforming images and masks together.
```python
# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_datagen.fit(images, augment=True, seed=seed)
mask_datagen.fit(masks, augment=True, seed=seed)
image_generator = image_datagen.flow_from_directory(
    'data/images',
    class_mode=None,
    seed=seed)
mask_generator = mask_datagen.flow_from_directory(
    'data/masks',
    class_mode=None,
    seed=seed)
# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)
model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)
```
Example of using `flow_from_dataframe(dataframe, directory, x_col, y_col)`:
```python
train_df = pandas.read_csv("./train.csv")
valid_df = pandas.read_csv("./valid.csv")
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='data/train',
        x_col="filename",
        y_col="class",
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
validation_generator = test_datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory='data/validation',
        x_col="filename",
        y_col="class",
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
```


----

<span style="float:right;">[[source]](www.dummy.com/my_project/tests/dummy_package/dummy_module.py#L340)</span>

### flow method


```python
ImageDataGenerator.flow(
    x,
    y=None,
    batch_size=32,
    shuffle=True,
    sample_weight=None,
    seed=None,
    save_to_dir=None,
    save_prefix="",
    save_format="png",
    subset=None,
)
```


Takes data & label arrays, generates batches of augmented data.

__Arguments__

- __x__: Input data. Numpy array of rank 4 or a tuple.
    If tuple, the first element
    should contain the images and the second element
    another numpy array or a list of numpy arrays
    that gets passed to the output
    without any modifications.
    Can be used to feed the model miscellaneous data
    along with the images.
    In case of grayscale data, the channels axis of the image array
    should have value 1, in case
    of RGB data, it should have value 3, and in case
    of RGBA data, it should have value 4.
- __y__: Labels.
- __batch_size__: Int (default: 32).
- __shuffle__: Boolean (default: True).
- __sample_weight__: Sample weights.
- __seed__: Int (default: None).
- __save_to_dir__: None or str (default: None).
    This allows you to optionally specify a directory
    to which to save the augmented pictures being generated
    (useful for visualizing what you are doing).
- __save_prefix__: Str (default: `''`).
    Prefix to use for filenames of saved pictures
    (only relevant if `save_to_dir` is set).
- __save_format__: one of "png", "jpeg"
    (only relevant if `save_to_dir` is set). Default: "png".
- __subset__: Subset of data (`"training"` or `"validation"`) if
    `validation_split` is set in `ImageDataGenerator`.

__Returns__

An `Iterator` yielding tuples of `(x, y)`
where `x` is a numpy array of image data
(in the case of a single image input) or a list
of numpy arrays (in the case with
additional inputs) and `y` is a numpy array
of corresponding labels. If 'sample_weight' is not None,
the yielded tuples are of the form `(x, y, sample_weight)`.
If `y` is None, only the numpy array `x` is returned.


----

<span style="float:right;">[[source]](www.dummy.com/my_project/tests/dummy_package/dummy_module.py#L395)</span>

### flow_from_directory method


```python
ImageDataGenerator.flow_from_directory(
    directory,
    target_size=(256, 256),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=None,
    save_to_dir=None,
    save_prefix="",
    save_format="png",
    follow_links=False,
    subset=None,
    interpolation="nearest",
)
```


Takes the path to a directory & generates batches of augmented data.

__Arguments__

- __directory__: string, path to the target directory.
    It should contain one subdirectory per class.
    Any PNG, JPG, BMP, PPM or TIF images
    inside each of the subdirectories directory tree
    will be included in the generator.
    See [this script](
    https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
    for more details.
- __target_size__: Tuple of integers `(height, width)`,
    default: `(256, 256)`.
    The dimensions to which all images found will be resized.
- __color_mode__: One of "grayscale", "rgb", "rgba". Default: "rgb".
    Whether the images will be converted to
    have 1, 3, or 4 channels.
- __classes__: Optional list of class subdirectories
    (e.g. `['dogs', 'cats']`). Default: None.
    If not provided, the list of classes will be automatically
    inferred from the subdirectory names/structure
    under `directory`, where each subdirectory will
    be treated as a different class
    (and the order of the classes, which will map to the label
    indices, will be alphanumeric).
    The dictionary containing the mapping from class names to class
    indices can be obtained via the attribute `class_indices`.
- __class_mode__: One of "categorical", "binary", "sparse",
    "input", or None. Default: "categorical".
    Determines the type of label arrays that are returned:
    - "categorical" will be 2D one-hot encoded labels,
    - "binary" will be 1D binary labels,
        "sparse" will be 1D integer labels,
    - "input" will be images identical
        to input images (mainly used to work with autoencoders).
    - If None, no labels are returned
      (the generator will only yield batches of image data,
      which is useful to use with `model.predict_generator()`).
      Please note that in case of class_mode None,
      the data still needs to reside in a subdirectory
      of `directory` for it to work correctly.
- __batch_size__: Size of the batches of data (default: 32).
- __shuffle__: Whether to shuffle the data (default: True)
    If set to False, sorts the data in alphanumeric order.
- __seed__: Optional random seed for shuffling and transformations.
- __save_to_dir__: None or str (default: None).
    This allows you to optionally specify
    a directory to which to save
    the augmented pictures being generated
    (useful for visualizing what you are doing).
- __save_prefix__: Str. Prefix to use for filenames of saved pictures
    (only relevant if `save_to_dir` is set).
- __save_format__: One of "png", "jpeg"
    (only relevant if `save_to_dir` is set). Default: "png".
- __follow_links__: Whether to follow symlinks inside
    class subdirectories (default: False).
- __subset__: Subset of data (`"training"` or `"validation"`) if
    `validation_split` is set in `ImageDataGenerator`.
- __interpolation__: Interpolation method used to
    resample the image if the
    target size is different from that of the loaded image.
    Supported methods are `"nearest"`, `"bilinear"`,
    and `"bicubic"`.
    If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
    supported. If PIL version 3.4.0 or newer is installed,
    `"box"` and `"hamming"` are also supported.
    By default, `"nearest"` is used.

__Returns__

A `DirectoryIterator` yielding tuples of `(x, y)`
where `x` is a numpy array containing a batch
of images with shape `(batch_size, *target_size, channels)`
and `y` is a numpy array of corresponding labels.


----

<span style="float:right;">[[source]](www.dummy.com/my_project/tests/dummy_package/dummy_module.py#L78)</span>

### to_categorical function


```python
tests.dummy_package.dummy_module.to_categorical(y, num_classes=None, dtype="float32")
```


Converts a class vector (integers) to binary class matrix.

E.g. for use with categorical_crossentropy.

__Arguments__

- __y__: class vector to be converted into a matrix
    (integers from 0 to num_classes).
- __num_classes__: total number of classes.
- __dtype__: The data type expected by the input, as a string
    (`float32`, `float64`, `int32`...)

__Returns__

A binary matrix representation of the input. The classes axis
is placed last.

__Example__

```python
# Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
> labels
array([0, 2, 1, 2, 0])
# `to_categorical` converts this into a matrix with as many
# columns as there are classes. The number of rows
# stays the same.
> to_categorical(labels)
array([[ 1.,  0.,  0.],
       [ 0.,  0.,  1.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.],
       [ 1.,  0.,  0.]], dtype=float32)
```


----

