import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from mnist import MNIST
from tensorflow import flags
from datetime import datetime

from library.utils import create_path, conv_layer
from library.utils import train_val_split, Dataset, BatchManager

FLAGS = flags.FLAGS

if __name__ == '__main__':
    flags.DEFINE_integer('num_classes', None,
                         'Number of classes in the target variable')
    flags.DEFINE_float('learning_rate', 1e-7,
                         'Learning rate')
    flags.DEFINE_integer('batch_size', 512,
                         'Number of date poitns in each training batch')
    flags.DEFINE_float('keep_prob', 0.6,
                         'Probability for each neuron to be kept')
    flags.DEFINE_integer('num_epochs', 200,
                         'Number of epochs for training')
    flags.DEFINE_boolean('shuffle', True,
                         'Whether shuffling the training set for each epoch')
    flags.DEFINE_integer('eval_frequency', 10,
                         'Number of steps between validation set '
                         'evaluations or model file updates')
    flags.DEFINE_integer('early_stopping_eval_rounds', 50,
                         'Perform early stop if the loss does '
                         'not drop in x evaluation rounds')
    flags.DEFINE_string('root_logdir', './tf_logs/',
                        'Root directory for storing tensorboard logs')
    flags.DEFINE_string('root_model_dir', './tf_models/',
                        'Root directory for storing tensorflow models')
    flags.DEFINE_integer('random_state', 666,
                         'Random state or seed')

def generate_log_model_dirs(root_logdir, root_model_dir):
    datetime_now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    logdir = root_logdir + 'run-{}/'.format(datetime_now)
    model_dir = root_model_dir + 'model-{}/'.format(datetime_now)
    create_path(model_dir)
    return logdir, model_dir

def generate_graph(logdir, learning_rate, num_classes, keep_prob, random_state):
    tf.reset_default_graph()
    graph = tf.Graph()

    with graph.as_default():
        img_inputs = tf.placeholder(tf.float32, [None, 28, 28, 1], 'img_inputs')
        img_labels = tf.placeholder(tf.int32, [None], 'img_labels')
        img_labels_onehot = tf.one_hot(
            img_labels, num_classes, axis=1, name='img_labels_onehot')

        with tf.variable_scope('cnn'):
            conv1 = conv_layer(
                'conv1', img_inputs, [3, 3], 16, max_pool=True,
                keep_prob=keep_prob, random_state=random_state)
            conv2 = conv_layer(
                'conv2', conv1, [3, 3], 32, max_pool=True,
                keep_prob=keep_prob, random_state=random_state)
            conv3 = conv_layer(
                'conv3', conv2, [3, 3], 64, max_pool=True,
                keep_prob=keep_prob, random_state=random_state)

        with tf.variable_scope('dense'):
            cnn_flatten = tf.layers.flatten(conv3, name='cnn_flatten')
            cnn_dense = tf.layers.dense(cnn_flatten, 1024, name='cnn_dense')

        with tf.variable_scope('output'):
            logit_out = tf.layers.dense(cnn_dense, num_classes, name='logit_out')
            softmax_out = tf.nn.softmax(logit_out, axis=1, name='softmax_out')

        with tf.variable_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=img_labels_onehot,
                logits=logit_out, name='loss')
            loss_avg = tf.reduce_mean(loss)

        with tf.variable_scope('optimization'):
            optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
            train = optimizer.minimize(loss_avg, name='optimizer')

        train_loss_summary = tf.summary.scalar(
            'Train_Loss', loss_avg)
        val_loss_summary = tf.summary.scalar(
            'Validation_Loss', loss_avg)
    file_writer = tf.summary.FileWriter(logdir, graph=graph)
    return graph, file_writer

def run_graph(graph, file_writer, model_dir, num_epochs, batch_size, shuffle, eval_freq,
              early_stopping_rounds, random_state):
    bst_score = np.inf
    step_counter = 1
    early_stopping_counter = 0

    tf.reset_default_graph()
    with graph.as_default():
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            img_inputs = graph.get_tensor_by_name('img_inputs:0')
            img_labels = graph.get_tensor_by_name('img_labels:0')
            optimizer = graph.get_operation_by_name('optimization/optimizer')
            loss_avg = graph.get_tensor_by_name('loss/Mean:0')
            train_loss_summary = graph.get_tensor_by_name('Train_Loss:0')
            val_loss_summary = graph.get_tensor_by_name('Validation_Loss:0')

            print('Loading data...')
            mndata = MNIST('./mnist_data/')
            images_train, labels_train = mndata.load_training()
            images_test, labels_test = mndata.load_testing()

            images_train = np.array(images_train)
            images_test = np.array(images_test)
            labels_train = np.array(labels_train.tolist())
            labels_test = np.array(labels_test.tolist())

            images_train = images_train.reshape([-1, 28, 28, 1])
            images_test = images_test.reshape([-1, 28, 28, 1])

            images_train, images_val, labels_train, labels_val = train_val_split(
                images_train, labels_train, random_state=random_state)
            train_set = Dataset(images_train, labels_train)
            val_set = Dataset(images_val, labels_val)
            test_set = Dataset(images_test, labels_test)
            batch_manager = BatchManager(train_set, num_epochs, shuffle, random_state)

            print('Training model...')
            while True:
                batch = batch_manager.next_batch(batch_size)
                if batch is None:
                    break
                batch_x, batch_y = batch[0], batch[1]

                if step_counter % eval_freq == 0:
                    train_loss = sess.run(loss_avg, feed_dict={
                        img_inputs:batch_x,
                        img_labels:batch_y
                    })

                    val_loss = sess.run(loss_avg, feed_dict={
                        img_inputs:val_set.X,
                        img_labels:val_set.y
                    })

                    print('Training Loss: {} | Validation Loss: {}'.format(
                        train_loss, val_loss))

                    summary_train_loss = sess.run(train_loss_summary, feed_dict={
                        img_inputs:batch_x,
                        img_labels:batch_y
                    })
                    file_writer.add_summary(summary_train_loss, step_counter)

                    summary_val_loss = sess.run(val_loss_summary, feed_dict={
                        img_inputs:val_set.X,
                        img_labels:val_set.y
                    })
                    file_writer.add_summary(summary_val_loss, step_counter)

                    if val_loss < bst_score:
                        early_stopping_counter = 0
                        saver.save(sess, model_dir+'gan.ckpt')
                    else:
                        early_stopping_counter += 1

                    if early_stopping_counter > early_stopping_rounds:
                        break

                sess.run(optimizer, feed_dict={
                    img_inputs:batch_x,
                    img_labels:batch_y
                })

                step_counter += 1

    sess.close()

def main(argv=None):
    logdir, model_dir = generate_log_model_dirs(
        FLAGS.root_logdir, FLAGS.root_model_dir)
    graph, file_writer = generate_graph(
        logdir, FLAGS.learning_rate, FLAGS.num_classes,
        FLAGS.keep_prob, FLAGS.random_state)
    run_graph(graph, file_writer, model_dir, FLAGS.num_epochs,
              FLAGS.batch_size, FLAGS.shuffle, FLAGS.eval_frequency,
              FLAGS.early_stopping_eval_rounds, FLAGS.random_state)

if __name__ == '__main__':
    tf.app.run()
