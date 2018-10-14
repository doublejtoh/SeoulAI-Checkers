import numpy as np
import tensorflow as tf




a = tf.constant([[8, 5], [3, 4]])

tf.InteractiveSession()
print (a.eval())
b = tf.reshape(a, [4])

c = tf.argmax(b, axis=0)
print(a)

col = tf.floormod(c, tf.int64(2))
row = tf.floordiv(c, tf.int64(2))

print(col, row)
