import os
import errno
import numpy as np
import tensorflow as tf

def create_path(path):
    """Create path if not exist"""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def conv_layer(name, input_tensor, ksize, num_out_channels, keep_prob, random_state,
               max_pool=False):
    input_channels = input_tensor.shape.as_list()[-1]
    conv_filter = tf.get_variable(
        '{}_filter'.format(name), [ksize[0], ksize[0], input_channels, num_out_channels],
        tf.float32, tf.truncated_normal_initializer)
    conv = tf.nn.conv2d(input_tensor, conv_filter, [1, 1, 1, 1], 'SAME', name=name)
    conv_relu = tf.nn.relu(conv, '{}_relu'.format(name))

    if max_pool:
        conv_maxpool = tf.nn.max_pool(
            conv_relu, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='{}_maxpool'.format(name))
        conv_dropout = tf.nn.dropout(conv_maxpool, keep_prob, seed=random_state, name='{}_dropout'.format(name))
        return conv_dropout

    conv_dropout = tf.nn.dropout(conv_relu, keep_prob, seed=random_state, name='{}_dropout'.format(name))
    return conv_dropout


def train_val_split(train_mtx, label_arr, random_state, train_proportion=0.8):
    np.random.seed(random_state)
    num_train_rows = np.round(train_mtx.shape[0] * train_proportion).astype(int)
    rows_selected = np.random.choice(train_mtx.shape[0],
                                     num_train_rows, replace=False)
    rows_not_selected = list(
        set(range(train_mtx.shape[0])) - set(rows_selected))

    return (train_mtx[rows_selected], train_mtx[rows_not_selected],
            label_arr[rows_selected], label_arr[rows_not_selected])


class Dataset():
    def __init__(self, X, y):
        self.X = X.copy()
        self.y = y.copy()


class BatchManager():

    def __init__(self, train_set, num_epochs, shuffle, random_state):
        """
        train_set, val_set: RNNDataset instances
        """
        self.train_set = train_set
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        self.random_state = random_state
        self.current_epoch = 0
        self.rows_in_batch = []

    def next_batch(self, batch_size):
        """
        Output next batch as (X, y), return None if ran over num_epochs
        """
        num_rows = self.train_set.X.shape[0]

        while len(self.rows_in_batch) < batch_size:
            self.current_epoch += 1
            row_nums = list(range(num_rows))
            if self.shuffle:
                np.random.seed(self.random_state)
                np.random.shuffle(row_nums)
            self.rows_in_batch += row_nums

        selected_X = self.train_set.X[self.rows_in_batch[:batch_size]]
        selected_y = self.train_set.y[self.rows_in_batch[:batch_size]]
        self.rows_in_batch = self.rows_in_batch[batch_size:]

        if self.current_epoch > self.num_epochs:
            return None
        return selected_X, selected_y
