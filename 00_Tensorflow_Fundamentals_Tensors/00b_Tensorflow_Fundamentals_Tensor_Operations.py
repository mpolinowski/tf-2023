import tensorflow as tf
import tensorflow_probability as tfp
# print(tf.__version__)
# 2.11.0
import numpy as np

# tensor with 4 dimensions
rank_4_tensor = tf.zeros(shape=[2, 3, 4, 5])
# get 2 elements of each dimension
print(rank_4_tensor[:2, :2, :2, :2])
# tf.Tensor(
# [[[[0. 0.]
#    [0. 0.]]

#   [[0. 0.]
#    [0. 0.]]]


#  [[[0. 0.]
#    [0. 0.]]

#   [[0. 0.]
#    [0. 0.]]]], shape=(2, 2, 2, 2), dtype=float32)

# get first element of all but the 4th dimension
print(rank_4_tensor[:1, :1, :1, :])
# tf.Tensor([[[[0. 0. 0. 0. 0.]]]], shape=(1, 1, 1, 5), dtype=float32)

# get first element of all but the 3rd dimension
print(rank_4_tensor[:1, :1, :, :1])
# tf.Tensor(
# [[[[0.]
#    [0.]
#    [0.]
#    [0.]]]], shape=(1, 1, 4, 1), dtype=float32)

# get first element of all but the 2nd dimension
print(rank_4_tensor[:1, :, :1, :1])
# tf.Tensor(
# [[[[0.]]

#   [[0.]]

#   [[0.]]]], shape=(1, 3, 1, 1), dtype=float32)


# get first element of all but the 1st dimension
print(rank_4_tensor[:, :1, :1, :1])
# tf.Tensor(
# [[[[0.]]]


#  [[[0.]]]], shape=(2, 1, 1, 1), dtype=float32)



# tensor with 2 dimensions
rank_2_tensor = tf.constant([[44, 88], [22, 77]])
print(rank_2_tensor.ndim)
# 2

# show last element of each row
print(rank_2_tensor[:, -1])
# tf.Tensor([88 77], shape=(2,), dtype=int32)


# add dimension to tensor
rank_3_tensor = rank_2_tensor[..., tf.newaxis]
# dot notation equals: rank_2_tensor[:, :, tf.newaxis]

print(rank_2_tensor)
# tf.Tensor(
# [[44 88]
#  [22 77]], shape=(2, 2), dtype=int32)

print(rank_3_tensor)
# tf.Tensor(
# [[[44]
#   [88]]

#  [[22]
#   [77]]], shape=(2, 2, 1), dtype=int32)


# Expand the final axis (-1)
rank_5_tensor = tf.expand_dims(rank_4_tensor, axis=-1)
print(rank_5_tensor.shape)
# (2, 3, 4, 5, 1)

# Expand the 0 axis
rank_5_tensor = tf.expand_dims(rank_4_tensor, axis=0)
print(rank_5_tensor.shape)
# (1, 2, 3, 4, 5)


# tensor manipulation
original_tensor = tf.constant([[44, 66], [33, 77]])

tensor_add = original_tensor + 4
print(tensor_add)
# tf.Tensor(
# [[48 70]
#  [37 81]], shape=(2, 2), dtype=int32)

tensor_subtract = original_tensor - 4
print(tensor_subtract)
# tf.Tensor(
# [[40 62]
#  [29 73]], shape=(2, 2), dtype=int32)

tensor_multiply = original_tensor * 99
print(tensor_multiply)
# tf.Tensor(
# [[4356 6534]
#  [3267 7623]], shape=(2, 2), dtype=int32)

print(tf.multiply(original_tensor, 99))
# tf.Tensor(
# [[4356 6534]
#  [3267 7623]], shape=(2, 2), dtype=int32)


# changing the datatype
## reducing precision 32 -> 16
tensor_float = tf.constant([9.13, 13.9])
print(tensor_float.dtype)
# <dtype: 'float32'>

tensor_float_16 = tf.cast(tensor_float, dtype=tf.float16)
print(tensor_float_16.dtype)
# <dtype: 'float16'>

## increasing precision
tensor_int = tf.constant([9, 14])
print(tensor_int)
# tf.Tensor([ 9 14], shape=(2,), dtype=int32)

tensor_int_to_float_16 = tf.cast(tensor_int, dtype=tf.float16)
print(tensor_int_to_float_16)
# tf.Tensor([ 9. 14.], shape=(2,), dtype=float16)

tensor_division = tf.divide(tensor_int_to_float_16, tensor_float_16)
print(tensor_division)
# tf.Tensor([0.9854 1.007 ], shape=(2,), dtype=float16)


# aggregation
tensor_random = tf.constant(np.random.randint(-100, 100, size=42))
print(tensor_random)
# tf.Tensor(
# [-32 -45  41 -23  -6 -66  90  63   1 -74 -66 -92  99  19 -97  94  50  51
#  -69 -60 -15  24  33  10 -86  74  92  55  95  30  91   7  44  55  35 -82
#   38  99  93 -92 -39  21], shape=(42,), dtype=int64)

## show absolute value
print(tf.abs(tensor_random))
# tf.Tensor(
# [32 45 41 23  6 66 90 63  1 74 66 92 99 19 97 94 50 51 69 60 15 24 33 10
#  86 74 92 55 95 30 91  7 44 55 35 82 38 99 93 92 39 21], shape=(42,), dtype=int64)

## show minimum value
print(tf.reduce_min(tensor_random))
# tf.Tensor(-97, shape=(), dtype=int64)

## show maximum value
print(tf.reduce_max(tensor_random))
# tf.Tensor(99, shape=(), dtype=int64)

## show mean value
print(tf.reduce_mean(tensor_random))
# tf.Tensor(10, shape=(), dtype=int64)

## show sum
print(tf.reduce_sum(tensor_random))
# tf.Tensor(460, shape=(), dtype=int64)


## calculate variance
print(tf.math.reduce_variance(tf.cast(tensor_random, dtype=tf.float16))) # needs type float
print(tfp.stats.variance(tensor_random))
# tf.Tensor(3921.0, shape=(), dtype=float16)
# tf.Tensor(3921, shape=(), dtype=int64)

## calculate standard deviation
print(tf.math.reduce_std(tf.cast(tensor_random, dtype=tf.float16))) # needs type float
print(tfp.stats.stddev(tf.cast(tensor_random, dtype=tf.float16))) # needs type float
# tf.Tensor(62.62, shape=(), dtype=float16)
# tf.Tensor(62.62, shape=(), dtype=float16)


# positional maximum & minimum
tf.random.set_seed(42)
tensor_const_seed = tf.random.uniform(shape=[42])
print(tensor_const_seed)
# tf.Tensor(
# [0.6645621  0.44100678 0.3528825  0.46448255 0.03366041 0.68467236
#  0.74011743 0.8724445  0.22632635 0.22319686 0.3103881  0.7223358
#  0.13318717 0.5480639  0.5746088  0.8996835  0.00946367 0.5212307
#  0.6345445  0.1993283  0.72942245 0.54583454 0.10756552 0.6767061
#  0.6602763  0.33695042 0.60141766 0.21062577 0.8527372  0.44062173
#  0.9485276  0.23752594 0.81179297 0.5263394  0.494308   0.21612847
#  0.8457197  0.8718841  0.3083862  0.6868038  0.23764038 0.7817228 ], shape=(42,), dtype=float32)

## find positional maximum
print(tf.reduce_max(tensor_const_seed))
print(tf.argmax(tensor_const_seed))
# tf.Tensor(0.9485276, shape=(), dtype=float32)
# tf.Tensor(30, shape=(), dtype=int64)

## find positional minimum
print(tf.reduce_min(tensor_const_seed))
print(tf.argmin(tensor_const_seed))
# tf.Tensor(0.009463668, shape=(), dtype=float32)
# tf.Tensor(16, shape=(), dtype=int64)


## find value by position index
print(tensor_const_seed[tf.argmax(tensor_const_seed)])
print(tensor_const_seed[tf.argmin(tensor_const_seed)])
# tf.Tensor(0.9485276, shape=(), dtype=float32)
# tf.Tensor(0.009463668, shape=(), dtype=float32