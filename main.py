import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Encoding ; load existing model's layers

    FCN-8 (aka 8 layers) encoder phase,
    using the VGG16 model (pre-trained on ImageNet for classification)
    to extract features.

    NOTE 1 : We will replace the VGG-16 fully-connected layers
    by 1-by-1 convolutions.

    NOTE 2 : The output from load_vgg() function will become the input to layers() function below.

    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    vgg_input_tensor_name = 'image_input:0'  # feed in an image

    # keep probability aka can 'defrost' frozen model weights to force model to learn more.
    vgg_keep_prob_tensor_name = 'keep_prob:0'  # how much data we want to throw away and avoid over-fitting

    vgg_layer3_out_tensor_name = 'layer3_out:0'  # convolution ouput
    vgg_layer4_out_tensor_name = 'layer4_out:0'  # convolution ouput
    vgg_layer7_out_tensor_name = 'layer7_out:0'  # convolution ouput

    vgg_tag = 'vgg16'

    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()  # gives access to all the layers

    # ************************************
    # Step 1 (see FCN architecture sketch)
    # ************************************
    # grab each layer by it's name
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)  # layer 1 aka the image
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)  # layer 2
    w3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)  # layer 3
    w4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)  # layer 4
    w7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)  # layer 7

    # return None, None, None, None, None
    return w1, keep, w3, w4, w7


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    1x1 Convolution and Decode

    Complete Encoding frozen vgg16 model by convolving Layer7 into a 1x1 convolution (instead of a fully
    connected layer) preserving the spatial information, then
    Skip Layers (Decode Step 2.1) via element-wise addition, then
    Upsample (deconvolving - Decode Step 2.2) the 1x1 convolution (aka Transpose Convolution or deconvolution)
    to the original sized image.

    Constructs Fully Convolutional neural Network FCN-8 (aka 8 layered) architecture.

    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output (size 4096)
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output (size 512)
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output (size 256)
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # ************************************
    # Step 2 (see FCN architecture sketch)
    # ************************************
    # convert output layers to 1x1 convolutional layers
    # tf.layers.conv2d() function params explanation :
    # vgg_layer7_out is furthest available layer in the frozen model,
    # num_classes (features) is binary classification ('filter') of is this pixel [road | not_road],
    # '1' is the kernel size aka to output 1x1 convolution layer,
    # use 'same' padding both here, and in tf.layers.conv2d_transpose() function below for size consistency,
    # kernel_regularizer= needed else during training your weights will become too large and prone to
    # overfitting and producing garbage aka penalises if weights become too large
    # conv_1x1_layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
    #                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(
    #                                       1e-3))  # with reguliser
    conv_1x1_layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same')  # without reguliser
    # conv_1x1_layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
    #                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(
    #                                       1e-3))  # with reguliser
    conv_1x1_layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same')  # without reguliser
    # conv_1x1_layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
    #                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(
    #                                       1e-3))  # with reguliser
    conv_1x1_layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same')  # without reguliser

    # ************************************
    # Step 3.1 (see FCN architecture sketch)
    # ************************************
    # up-sample by '2' and add 1st skip layer
    output = tf.layers.conv2d_transpose(conv_1x1_layer7, num_classes, 4, 2, 'same')  # scale up by x2
    output = tf.add(output, conv_1x1_layer4)  # 1st skip layer

    # ************************************
    # Step 3.2 (see FCN architecture sketch)
    # ************************************
    # up-sample by '2' and add 2nd skip layer
    # Deconvolution layer 'output'
    # tf.layers.conv2d_transpose() function params explanation :
    # conv_1x1 layer (iterator) containing image spatial information,
    # num_classes (features) is binary classification ('filter') of is this pixel [road | not_road],
    # '4' is the kernel size,
    # '2' strides which will cause the kernel output to be upsampled by 2,
    # use 'same' padding both here, and in tf.layers.conv2d() function above for size consistency,
    # kernel_regularizer= needed else during training your weights will become too large and prone to
    # overfitting and producing garbage aka penalises if weights become too large
    # output = tf.layers.conv2d_transpose(output, num_classes, 4, 2, padding='same',
    #                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output = tf.layers.conv2d_transpose(output, num_classes, 4, 2, 'same')  # without regulariser
    output = tf.add(output, conv_1x1_layer3)  # 2nd skip layer

    # ************************************
    # Step 4 (see FCN architecture sketch)
    # ************************************
    # up-sample by '8' to original image size
    output = tf.layers.conv2d_transpose(output, num_classes, 16, 8,
                                        'same')  # scale up by x8 to get original image size

    # return None
    return output  # a 4D tensor


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Define a loss measure so we can approach training a FCN like a normal classification CNN.

    Goal : assign each pixel to an appropriate class, in our case [road | not_road].

    Reshape the 4D tensor output from layers() function above into a 2D sensor of logits, where
    rows are pixels and columns are the 2 classes.

    We can then apply the cross entropy loss function.

    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    # return None, None, None
    return logits, optimizer, cross_entropy_loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    for epoch in range(epochs):
        for batch, (image, label) in enumerate(get_batches_fn(batch_size)):
            feed_dict = {input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 1e-5}
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
            print('Epoch ', epoch, ' Batch ', batch, ' Loss ', loss)

    pass


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        epochs = 6  # road / not_road spread evenly across image aka garbage results!
        #epochs = 55  # will try more training after this git push
        batch_size = 2
        learning_rate = tf.placeholder(tf.float32)
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])

        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, optimizer, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate,
                                                         num_classes)

        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, optimizer, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
