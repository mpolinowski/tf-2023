import tensorflow as tf
# print(tf.__version__)
# 2.11.0
import numpy as np

# create tensors with tf.constant()
scalar = tf.constant(88, name='scalar')
print(scalar)
# tf.Tensor(88, shape=(), dtype=int32)
print(scalar.ndim)
# 0 dimensions

vector = tf.constant([44, 88], name='vector')
print(vector)
# tf.Tensor([44 88], shape=(2,), dtype=int32)
print(vector.ndim)
# 1 dimensions

matrix = tf.constant([[44., 88.], [33., 55.]], shape=(2, 2), dtype=tf.float16, name='matrix')
print(matrix)
# tf.Tensor(
# [[44. 88.]
#  [33. 55.]], shape=(2, 2), dtype=float16)
print(matrix.ndim)
# 2 dimensions

tensor = tf.constant([[[44, 88, 22, 66],
                     [666, 222, 999, 333]],
                     [[33, 11, 55, 77],
                     [111, 888, 444, 111]]], shape=(4, 2, 2), dtype=tf.int16, name='tensor')
print(tensor)
# tf.Tensor(
# [[[ 44  88]
#   [ 22  66]]

#  [[666 222]
#   [999 333]]

#  [[ 33  11]
#   [ 55  77]]

#  [[111 888]
#   [444 111]]], shape=(4, 2, 2), dtype=int16)
print(tensor.ndim)
# 3 dimensions



# create tensors with tf.Variable()
constant_tensor = tf.constant([44, 88], name='constant')
print(constant_tensor)
# tf.Tensor([44 88], shape=(2,), dtype=int32)

variable_tensor = tf.Variable([44, 88], name='variable')
print(variable_tensor)
# <tf.Variable 'variable:0' shape=(2,) dtype=int32, numpy=array([44, 88], dtype=int32)>

## change values in tensor
variable_tensor[0].assign(77)
print(variable_tensor)
# <tf.Variable 'variable:0' shape=(2,) dtype=int32, numpy=array([77, 88], dtype=int32)>

# constant_tensor[0].assign(77)
# AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'assign'



# create random tensors with tf.random()
## fixed seed for reproducibility
random_tensor_normal = tf.random.Generator.from_seed(42)
## Output 2x2 matrix of random values from a normal distribution
random_tensor_normal = random_tensor_normal.normal(shape=(2, 2))
print(random_tensor_normal)
# tf.Tensor(
# [[-0.7565803  -0.06854702]
#  [ 0.07595026 -1.2573844 ]], shape=(2, 2), dtype=float32)


## fixed seed for reproducibility
random_tensor_uniform = tf.random.Generator.from_seed(42)
## Output 2x2 matrix of random values from a uniform distribution
random_tensor_uniform = random_tensor_uniform.uniform(shape=(2, 2))
print(random_tensor_uniform)
# tf.Tensor(
# [[0.7493447  0.73561966]
#  [0.45230794 0.49039817]], shape=(2, 2), dtype=float32)


## prove pseudo-randomness
random_tensor_1 = tf.random.Generator.from_seed(42)
random_tensor_1 = random_tensor_1.normal(shape=(2, 2))
random_tensor_2 = tf.random.Generator.from_seed(42)
random_tensor_2 = random_tensor_2.normal(shape=(2, 2))
print(random_tensor_1 == random_tensor_2)
# tf.Tensor(
# [[ True  True]
#  [ True  True]], shape=(2, 2), dtype=bool)


## shuffle order of generated values
constant_matrix = tf.constant([[44, 88],
                       [77, 55],
                       [1, 3]], name='constant')

print(constant_matrix)
# tf.Tensor(
# [[44 88]
#  [77 55]
#  [ 1  3]], shape=(3, 2), dtype=int32)

## shuffle derives seed from both global and function level
## you have to set both to get the same shuffle on every run
## tf.random.set_seed(42)
shuffled_matrix = tf.random.shuffle(constant_matrix, seed=42, name='shuffled')
                       
print(shuffled_matrix)
# tf.Tensor(
# [[ 1  3]
#  [77 55]
#  [44 88]], shape=(3, 2), dtype=int32)


# creating tensors with numpy
## return tensors with `1` values
tensor_one = tf.ones([3, 4], dtype=tf.int16)
print(tensor_one)
# tf.Tensor(
# [[1 1 1 1]
#  [1 1 1 1]
#  [1 1 1 1]], shape=(3, 4), dtype=int16)

## return tensors with `0` values
tensor_zero = tf.zeros([3, 4], dtype=tf.int16)
print(tensor_zero)
# tf.Tensor(
# [[0 0 0 0]
#  [0 0 0 0]
#  [0 0 0 0]], shape=(3, 4), dtype=int16)

## turn numpy array into tensor
numpy_array = np.arange(1, 25, dtype=np.int16)
print(numpy_array)
# [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]

tf_vector = tf.constant(numpy_array)
print(tf_vector)
# tf.Tensor([ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24], shape=(24,), dtype=int16)

tf_tensor = tf.constant(numpy_array, shape=(2, 3, 4))
print(tf_tensor)
# tf.Tensor(
# [[[ 1  2  3  4]
#   [ 5  6  7  8]
#   [ 9 10 11 12]]

#  [[13 14 15 16]
#   [17 18 19 20]
#   [21 22 23 24]]], shape=(2, 3, 4), dtype=int16)


# getting tensor attributes
## rank
### number of tensor dimensions
### example: create a rank 4 tensor
rank_4_tensor = tf.zeros(shape=[2, 2, 2, 2])
print(rank_4_tensor)
# tf.Tensor(
# [[[[0. 0.]
#    [0. 0.]]

#   [[0. 0.]
#    [0. 0.]]]


#  [[[0. 0.]
#    [0. 0.]]

#   [[0. 0.]
#    [0. 0.]]]], shape=(2, 2, 2, 2), dtype=float32)
print(rank_4_tensor.ndim)
# 4

## shape
### number of elements of each dimension
print(rank_4_tensor.shape)
# (2, 2, 2, 2)

## axis
### a selected dimension
print(rank_4_tensor[0])
# tf.Tensor(
# [[[0. 0.]
#   [0. 0.]]

#  [[0. 0.]
#   [0. 0.]]], shape=(2, 2, 2), dtype=float32)

## size
### total number of items
print(tf.size(rank_4_tensor))
# tf.Tensor(16, shape=(), dtype=int32)

## bringing it all together:
print("INFO :: Datatype of every element:", rank_4_tensor.dtype)
print("INFO :: Number of dimensions (Rank):", rank_4_tensor.ndim)
print("INFO :: Number of Elements in Tensor:", tf.size(rank_4_tensor).numpy())
print("INFO :: Tensor shape:", rank_4_tensor.shape)
print("INFO :: Elements along 0 Axis:", rank_4_tensor.shape[0])
print("INFO :: Elements along last Axis:", rank_4_tensor.shape[-1])

# INFO :: Datatype of every element: <dtype: 'float32'>
# INFO :: Number of dimensions (Rank): 4
# INFO :: Number of Elements in Tensor: 16
# INFO :: Tensor shape: (2, 2, 2, 2)
# INFO :: Elements along 0 Axis: 2
# INFO :: Elements along last Axis: 2