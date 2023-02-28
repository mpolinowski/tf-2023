import tensorflow as tf
# print(tf.__version__)
# 2.11.0
import numpy as np

tensor = tf.constant([[2, 4],
                       [3, 5]])

print(tensor)
# tf.Tensor(
# [[2 4]
#  [3 5]], shape=(2, 2), dtype=int32)

# matrix multiplications
result = tf.matmul(tensor, tensor)
print(result)
# tf.Tensor(
# [[16 28]
#  [21 37]], shape=(2, 2), dtype=int32)



# multiplying matrices with mis-matched shape
tensor1 = tf.constant([[1, 2],
                        [3, 4],
                        [5, 6]])

tensor2 = tf.constant([[1, 2],
                        [3, 4],
                        [5, 6]])

# matrix multiplications
# result = tf.matmul(tensor1, tensor2)
# Matrix size-incompatible: In[0]: [3,2], In[1]: [3,2] [Op:MatMul]


# reshape `tensor2` to make them compatible:
tensor3 = tf.reshape(tensor2, shape=(2, 3))
print(tensor3)
# tf.Tensor(
# [[1 2 3]
#  [4 5 6]], shape=(2, 3), dtype=int32)

# and try again
result = tf.matmul(tensor1, tensor3)
print(result)
# tf.Tensor(
# [[ 9 12 15]
#  [19 26 33]
#  [29 40 51]], shape=(3, 3), dtype=int32)


# reshape `tensor1` to make them compatible:
tensor4 = tf.reshape(tensor1, shape=(2, 3))
print(tensor4)
# tf.Tensor(
# [[1 2 3]
#  [4 5 6]], shape=(2, 3), dtype=int32)

# and try again
result = tf.matmul(tensor4, tensor2)
print(result)
# tf.Tensor(
# [[22 28]
#  [49 64]], shape=(2, 2), dtype=int32)


# transpose - instead of re-shape -`tensor1` to make them compatible:
tensor5 = tf.transpose(tensor1)
print(tensor5)
# tf.Tensor(
# [[1 3 5]
#  [2 4 6]], shape=(2, 3), dtype=int32)

# and try again
result = tf.matmul(tensor5, tensor2)
print(result)
# tf.Tensor(
# [[35 44]
#  [44 56]], shape=(2, 2), dtype=int32)


# dot-product
result = tf.tensordot(tensor5, tensor2, axes=1)
print(result)
# tf.Tensor(
# [[35 44]
#  [44 56]], shape=(2, 2), dtype=int32)


# squeezing
tf.random.set_seed(42)
## create a tensor with additional dimensions
tensor_unsqueezed = tf.constant(tf.random.uniform(shape=[42]), shape=[1, 1, 1, 1, 42])
print(tensor_unsqueezed)
# tf.Tensor(
# [[[[[0.6645621  0.44100678 0.3528825  0.46448255 0.03366041 0.68467236
#      0.74011743 0.8724445  0.22632635 0.22319686 0.3103881  0.7223358
#      0.13318717 0.5480639  0.5746088  0.8996835  0.00946367 0.5212307
#      0.6345445  0.1993283  0.72942245 0.54583454 0.10756552 0.6767061
#      0.6602763  0.33695042 0.60141766 0.21062577 0.8527372  0.44062173
#      0.9485276  0.23752594 0.81179297 0.5263394  0.494308   0.21612847
#      0.8457197  0.8718841  0.3083862  0.6868038  0.23764038 0.7817228 ]]]]], shape=(1, 1, 1, 1, 42), dtype=float32)

# remove all dimensions of size 1
tensor_squeezed = tf.squeeze(tensor_unsqueezed)
print(tensor_squeezed)
# tf.Tensor(
# [0.6645621  0.44100678 0.3528825  0.46448255 0.03366041 0.68467236
#  0.74011743 0.8724445  0.22632635 0.22319686 0.3103881  0.7223358
#  0.13318717 0.5480639  0.5746088  0.8996835  0.00946367 0.5212307
#  0.6345445  0.1993283  0.72942245 0.54583454 0.10756552 0.6767061
#  0.6602763  0.33695042 0.60141766 0.21062577 0.8527372  0.44062173
#  0.9485276  0.23752594 0.81179297 0.5263394  0.494308   0.21612847
#  0.8457197  0.8718841  0.3083862  0.6868038  0.23764038 0.7817228 ], shape=(42,), dtype=float32)


# one-hot encoding
## create a list
a_list = [0, 1, 2, 3, 4]
## encode
print(tf.one_hot(a_list, depth=5))
# tf.Tensor(
# [[1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 1. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1.]], shape=(5, 5), dtype=float32)


## encode and assign values
print(tf.one_hot(a_list, depth=5, on_value=100, off_value=-100))
# tf.Tensor(
# [[ 100 -100 -100 -100 -100]
#  [-100  100 -100 -100 -100]
#  [-100 -100  100 -100 -100]
#  [-100 -100 -100  100 -100]
#  [-100 -100 -100 -100  100]], shape=(5, 5), dtype=int32)



# square, square root and log
tensor_not_squared = tf.range(1, 10)
print(tensor_not_squared)
# tf.Tensor([1 2 3 4 5 6 7 8 9], shape=(9,), dtype=int32)

# square all values
tensor_squared = tf.square(tensor_not_squared)
print(tensor_squared)
# tf.Tensor([ 1  4  9 16 25 36 49 64 81], shape=(9,), dtype=int32)

# square all values
tensor_unsquared = tf.sqrt(tf.cast(tensor_squared, dtype="float16")) # needs non-int type
print(tensor_unsquared)
# tf.Tensor([1. 2. 3. 4. 5. 6. 7. 8. 9.], shape=(9,), dtype=float16)

# get log
tensor_log = tf.math.log(tf.cast(tensor_unsquared, dtype="float16")) # needs float
print(tensor_log)
# tf.Tensor([0.     0.6934 1.099  1.387  1.609  1.792  1.946  2.08   2.197 ], shape=(9,), dtype=float16)



# numpy arrays
## numpy array -> tensorflow tensor
tf_np_array = tf.constant(np.array([1., 2., 3., 4., 5.]))
print(tf_np_array)
# tf.Tensor([1. 2. 3. 4. 5.], shape=(5,), dtype=float64)

## tensorflow tensor -> numpy array
np_tf_tensor = np.array(tf_np_array)
print(np_tf_tensor, type(np_tf_tensor))
# [1. 2. 3. 4. 5.] <class 'numpy.ndarray'>

## tensorflow tensor -> numpy array
## just a different method
np_tf_tensor2 = tf_np_array.numpy()
print(np_tf_tensor2, type(np_tf_tensor2))
# [1. 2. 3. 4. 5.] <class 'numpy.ndarray'>


## extract a single value from a tensor
my_value = tf_np_array.numpy()[0]
print(my_value)
# 1.0


## different default types!
numpy_tensor = tf.constant(np.array([1., 2., 3., 4., 5.]))
print(numpy_tensor)
# tf.Tensor([1. 2. 3. 4. 5.], shape=(5,), dtype=float64)

tf_tensor = tf.constant([1., 2., 3., 4., 5.])
print(tf_tensor)
# tf.Tensor([1. 2. 3. 4. 5.], shape=(5,), dtype=float32)


# make sure you have GPU/TPU support
print(tf.config.list_physical_devices())
# [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Num GPUs Available:  1