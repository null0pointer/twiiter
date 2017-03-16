import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
import tfutils as tfu

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!@#$^&*()_-=| "
ALPHABET_LENGTH = len(ALPHABET)
SEQUENCE_LENGTH = 30
BATCHSIZE = 100
INTERNAL_SIZE = 512
LAYERS = 3
INITIAL_LEARNING_RATE = 0.001
DROPOUT_KEEP_RATE = 1.0

lr = tf.placeholder(tf.float32, name='lr')
pkeep = tf.placeholder(tf.float32, name='pkeep')
batchsize = tf.placeholder(tf.int32, name='batchsize')

# Inputs
X = tf.placeholder(tf.uint32, [None, None], name='X') # [ BATCHSIZE, SEQUENCE_LENGTH ]
Xo = tf.one_hot(X, ALPHABET_LENGTH, 1.0, 0.0) # [ BATCHSIZE, SEQUENCE_LENGTH , ALPHABET_LENGTH ]

# Expected outputs
Y_ = tf.placeholder(tf.uint8, [None, None], name='Y_') # [ BATCHSIZE, SEQUENCE_LENGTH ]
Yo_ = tf.one_hot(Y_, ALPHABET_LENGTH, 1.0, 0.0) # [ BATCHSIZE, SEQUENCE_LENGTH , ALPHABET_LENGTH ]

# Initial internal cell state
Hin = tf.placeholder(tf.float32, [None, INTERNAL_SIZE * LAYERS], name='Hin') # [ BATCHSIZE , INTERNAL_SIZE * LAYERS ]

# Deep stacked GRU cell
deep_drop_cell = tfu.rnn.MultiDropoutGRUCell(size=INTERNAL_SIZE, pkeep=DROPOUT_KEEP_RATE, layers=LAYERS)

# Output predictions and output state
Yr, H = tf.nn.dynamic_rnn(deep_drop_cell, Xo, dtype=tf.float32, initial_state=Hin)

H = tf.identity(H, name='H')

Yflat = tf.reshape(Yr, [-1, INTERNAL_SIZE])
Ylogits = layers.linear(Yflat, ALPHABET_LENGTH)
Yflat_ = tf.reshape(Yo_, [-1, ALPHABET_LENGTH])
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)
loss = tf.reshape(loss, [batchsize, -1])
Yo = tf.nn.softmax(Ylogits, name='Yo')
Y = tf.argmax(Yo, 1)
Y = tf.reshape(Y, [batchsize, -1], name='Y')
train_step = tf.train.AdamOptimizer(lr).minimize(loss)