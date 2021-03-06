import os.path
import os
import shutil
import tensorflow as tf
from keras import backend as K
from keras.utils import plot_model
import helper
import warnings
from distutils.version import LooseVersion
from seg_mobilenet import SegMobileNet
import project_tests as tests
from tqdm import tqdm
import numpy as np
from IPython import embed
from augmentation import rotate_both, flip_both, blur_both, illumination_change_both  # noqa
import glob
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# https://keras.io/backend/
KERAS_TRAIN = 1
KERAS_TEST = 0
use_ce_loss = False
epsilon = 1e-5

from common import *

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), \
    'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(
        tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn(
        'No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def calc_f(beta, pr, rc):
    f_val = ((1 + beta ** 2) * pr * rc) / (epsilon + ((beta ** 2) * pr) + rc)
    return f_val


def calc_fmax(beta):
    return (1 + beta ** 2) / (1 + beta)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    if use_ce_loss:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits))
    else:
        logits_c = tf.unstack(logits, axis=1)
        correct_label_c = tf.unstack(correct_label, axis=1)
        loss = 0
        beta_car = 2.0
        beta_road = 0.5
        weights = [0.0, 0.8, 0.2]  # Background, Car, Road
        F_car = 0
        F_road = 0
        for i in range(num_classes):
            # Ignore background
            if i == BG_ID:
                continue
            l, c = logits_c[i], correct_label_c[i]
            inter = tf.reduce_sum(tf.multiply(l, c))
            union = tf.reduce_sum(tf.subtract(tf.add(l, c), tf.multiply(l, c)))
            precision = tf.divide(inter, tf.reduce_sum(l))
            recall = tf.divide(inter, tf.reduce_sum(c))
            if i == CAR_ID:
                F_car = calc_f(beta_car, precision, recall)
            if i == ROAD_ID:
                F_road = calc_f(beta_road, precision, recall)
            # loss+=weights[i]*tf.subtract(tf.constant(1.0, dtype=tf.float32),tf.divide(inter,union))
        # loss=tf.subtract(tf.constant(1.0, dtype=tf.float32),(F_car + F_road))
        F_max = weights[1] * calc_fmax(beta_car) + weights[2] * calc_fmax(beta_road)
        ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits))
        # loss = (F_max - (weights[1]*F_car + weights[2]*F_road))
        loss = ce_loss + (F_max - (weights[1] * F_car + weights[2] * F_road))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    train_op = optimizer.minimize(loss)

    predicted_label = tf.argmax(logits, axis=-1)
    sparse_correct_label = tf.argmax(correct_label, axis=-1)
    with tf.variable_scope("iou") as scope:
        iou, iou_op = tf.metrics.mean_iou(
            sparse_correct_label, predicted_label, num_classes)
    metric_vars = [v for v in tf.local_variables()
                   if v.name.split('/')[0] == 'iou']
    metric_reset_ops = tf.variables_initializer(metric_vars)
    return logits, train_op, loss, ce_loss, iou, iou_op, metric_reset_ops


# tests.test_optimize(optimize)


def train_nn(
        sess,
        epochs,
        batch_size,
        train_batches_fn, val_batches_fn,
        train_op,
        total_loss, cross_entropy_loss,
        input_image, correct_label,
        learning_rate, learning_rate_val, decay, learning_phase,
        iou_op, iou,
        metric_reset_ops, update_ops,
        model, N_train, N_val):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.
           Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    train_loss_summary = tf.placeholder(tf.float32)
    train_ce_loss_summary = tf.placeholder(tf.float32)
    val_loss_summary = tf.placeholder(tf.float32)
    val_ce_loss_summary = tf.placeholder(tf.float32)
    train_iou_summary = tf.placeholder(tf.float32)
    val_iou_summary = tf.placeholder(tf.float32)
    seg_image_summary = tf.placeholder(tf.float32, [None, nw_shape[0], nw_shape[1], 3], name="seg_img")
    inter_seg_image_summary = tf.placeholder(tf.float32, [None, nw_shape[0], nw_shape[1], 3], name="inter_seg_img")

    tf.summary.scalar("train_loss", train_loss_summary)
    tf.summary.scalar("train_ce_loss", train_ce_loss_summary)
    tf.summary.scalar("train_iou", train_iou_summary)
    tf.summary.scalar("val_loss", val_loss_summary)
    tf.summary.scalar("val_ce_loss", val_ce_loss_summary)
    tf.summary.scalar("val_iou", val_iou_summary)
    tf.summary.scalar("learning_rate", learning_rate)
    tf.summary.image('input_img', input_image)
    seg_summary = tf.summary.image('seg_img', seg_image_summary)
    inter_seg_summary = tf.summary.image('inter_seg_img', inter_seg_image_summary)

    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('log', graph=sess.graph)

    all_vars = tf.global_variables()
    optimizer_variables = [v for v in all_vars
                           if v not in model.updates and
                           v not in model.trainable_weights]

    # embed()
    sess.run(metric_reset_ops)
    sess.run(tf.variables_initializer(optimizer_variables))
    for var in all_vars:
        try:
            sess.run(var)
        except:
            print("Err ", var)
            embed()
    epoch_pbar = tqdm(range(epochs), ncols=50)
    for epoch in epoch_pbar:
        # train
        train_loss = 0.0
        train_ce_loss = 0.0
        iteration_counter = 0
        for image, label in train_batches_fn(batch_size):
            fetches = [total_loss, cross_entropy_loss, train_op, iou_op] + update_ops
            feed_dict = {
                input_image: image, correct_label: label,
                learning_rate: learning_rate_val, learning_phase: KERAS_TRAIN,
            }
            loss_val, ce_loss_val, *_ = sess.run(  # noqa
                fetches,
                feed_dict=feed_dict)
            # embed()
            train_loss += loss_val
            train_ce_loss += ce_loss_val
            iteration_counter += 1

            if iteration_counter % 100 == 0:
                segmented_images = []
                for i in range(5):
                    img1 = helper.get_seg_img(sess, model.output, input_image, image[i], image_shape=nw_shape,
                                              nw_shape=nw_shape, learning_phase=learning_phase)
                    arg_label = label[i].argmax(axis=2)
                    img2 = helper.blend_output(img1, arg_label, (255, 5, 0), (5, 5, 5), nw_shape)
                    segmented_images.append(np.array(img2))
                summary_val = sess.run(inter_seg_summary, feed_dict={inter_seg_image_summary: segmented_images})
                writer.add_summary(summary_val, int(iteration_counter))

        learning_rate_val = learning_rate_val / (1.0 + decay * epoch)
        train_iou = sess.run(iou)
        train_loss /= iteration_counter
        train_ce_loss /= iteration_counter
        val_ce_loss = 0.0
        val_loss = 0.0
        iteration_counter = 0
        sess.run(metric_reset_ops)
        # val
        for image, label in val_batches_fn(batch_size):
            feed_dict = {
                input_image: image[:, -nw_shape[0]:, :], correct_label: label[:, -nw_shape[0]:, :],
                learning_phase: KERAS_TEST,
            }
            *_, loss_val, ce_loss_val = sess.run(
                [iou_op, total_loss, cross_entropy_loss], feed_dict=feed_dict)
            val_loss += loss_val
            val_ce_loss += ce_loss_val
            iteration_counter += 1

        segmented_images = []
        for i in range(5):
            segmented_images.append(np.array(helper.get_seg_img(sess, model.output, input_image, image[i],
                                                                image_shape=nw_shape, nw_shape=nw_shape,
                                                                learning_phase=learning_phase)))

        val_iou = sess.run(iou)
        val_loss /= iteration_counter
        val_ce_loss /= iteration_counter
        epoch_pbar.write(
            "Epoch %03d: loss: %.4f ce_loss: %.4f mIoU: %.4f val_loss: %.4f val_ce_loss: %.4f val_mIoU: %.4f"
            % (epoch, train_loss, train_ce_loss, train_iou, val_loss, val_ce_loss, val_iou))
        summary_val = sess.run(
            summary, feed_dict={train_loss_summary: train_loss,
                                train_ce_loss_summary: train_ce_loss,
                                val_loss_summary: val_loss,
                                val_ce_loss_summary: val_ce_loss,
                                train_iou_summary: train_iou,
                                val_iou_summary: val_iou,
                                learning_rate: learning_rate_val,
                                input_image: image,
                                seg_image_summary: np.array(segmented_images),
                                inter_seg_image_summary: np.array(segmented_images)})

        writer.add_summary(summary_val, epoch)
        if epoch % 1 == 0:
            weight_path = 'checkpoint/ep-newweight-%03d-val_loss-%.4f.hdf5' \
                          % (epoch, val_loss)
            model.save(weight_path)


def augmentation_fn(image, label, label2, label3):
    """Wrapper for augmentation methods
    """
    image = np.uint8(image)
    label = np.uint8(label)
    label2 = np.uint8(label2)
    label3 = np.uint8(label3)
    image, label, label2, label3 = flip_both(image, label, label2, label3, p=0.5)
    # image, label, label2, label3 = rotate_both(image, label, label2, label3, p=0.5, ignore_label=1)
    # image, label, label2, label3 = blur_both(image, label, label2, label3, p=0.5)
    image, label, label2, label3 = illumination_change_both(image, label, label2, label3, p=0.5)
    return image, label == 1, label2 == 1, label3 == 1


# tests.test_train_nn(train_nn)
def run():
    from_scratch = False
    do_train = False
    learning_rate_val = 0.001
    epochs = 20
    decay = learning_rate_val / (2 * epochs)
    batch_size = 12
    data_dir = './Train'
    runs_dir = './runs'

    if not from_scratch:
        weight_path = helper.maybe_download_mobilenet_weights()

    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')

    model_files = glob.glob('checkpoint/ep-*.hdf5')
    model_files.sort(key=os.path.getmtime)
    print(model_files[-1])
    with K.get_session() as sess:
        # Create function to get batches
        # train_batches_fn, val_batches_fn = helper.gen_batches_functions(
        # os.path.join(data_dir, 'data_road/training'), image_shape,
        # train_augmentation_fn=augmentation_fn)
        train_batches_fn, val_batches_fn, N_train, N_val = helper.gen_lyft_batches_functions(
            data_dir, image_shape, nw_shape, image_folder='CameraRGB', label_folder='CameraSeg',
            train_augmentation_fn=augmentation_fn)

        learning_phase = K.learning_phase()
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        correct_label = tf.placeholder(
            tf.float32,
            shape=[None, nw_shape[0], nw_shape[1], n_classes],
            name='correct_label')

        model = SegMobileNet(nw_shape[0], nw_shape[1], num_classes=n_classes)
        # this initializes the keras variables
        sess = K.get_session()
        if not from_scratch:
            # model.load_weights(weight_path, by_name=True)
            model.load_weights(model_files[-1])
        input_image = model.input
        logits = model.output
        # https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html  # noqa
        update_ops = model.updates

        logits, train_op, loss, ce_loss, iou, iou_op, metric_reset_ops = optimize(
            logits, correct_label, learning_rate, n_classes)

        if do_train:
            print("N_train: %d\tN_val:%d\tTrain steps: %d\tVal_steps: %d\tBatch size: %d" % (
                N_train, N_val, int(N_train / batch_size), int(N_val / batch_size), batch_size))
            train_nn(sess, epochs, batch_size,
                     train_batches_fn, val_batches_fn,
                     train_op, loss, ce_loss, input_image,
                     correct_label,
                     learning_rate, learning_rate_val, decay, learning_phase,
                     iou_op, iou, metric_reset_ops,
                     update_ops,
                     model, N_train, N_val)
        else:
            model.load_weights(model_files[-1])
        plot_model(model, to_file='seg_MobileNetModel.png')
        helper.lyft_save_inference_samples(
            runs_dir, data_dir, 'CameraRGB', sess, image_shape, nw_shape,
            logits, learning_phase, input_image)



if __name__ == '__main__':
    run()
