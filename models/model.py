import sys
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet import ResNet50
from .wrn import wide_residual_network
from .resnet import ResNet18, ResNet34
from .resnet_cifar10_v2 import get_resnet_v2_model
import tensorflow_addons as tfa
from losses.Triplet import triplet_loss


def adaptive_gpu_memory(gpu_id="0"):
    """
    Helper function for restricting model to occupy only required GPU memory.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def get_network(arch, input_shape=(32, 32, 3), weights=None, pool='avg'):
    """
    Returns base model
    :param arch: network architecture
    :param input_shape: shapes of input dataset
    :param weights: use pretrained weights or not
    :param pool: use pretrained weights or not
    :return: tf.keras model
    """
    # adaptive_gpu_memory()
    if 'vgg16' in arch:
        es = 256
        conv_base = VGG16(input_shape=input_shape, include_top=False, weights=weights, pooling='avg')
    elif 'wrn' in arch:
        es = 64
        dw = arch.split('-')[1:]
        conv_base = wide_residual_network(depth=int(dw[0]), width=int(dw[1]), input_shape=input_shape, weights=weights,
                                          pool=pool)
    elif 'resnet18' in arch:
        es = 64
        conv_base = ResNet18()  #
    elif 'resnet34' in arch:
        es = 256
        conv_base = ResNet34()  #
    elif 'resnet20' in arch:
        es = 128
        conv_base = get_resnet_v2_model(name="resnet20", size=input_shape[0], include_top=False)
    elif 'resnet50' in arch:
        es = 256
        conv_base = ResNet50(input_shape=input_shape, include_top=False, weights=weights, pooling='avg')
    else:
        print(arch, " : not implemented")
        sys.exit(0)

    return conv_base, es


def get_optimizer(opt, lr):
    """
    Creates an tf.keras.optimizer object
    :param opt: name of optimizer: adam, sgd, and rmsprop
    :param lr: learning rate
    :return: optimizer
    """
    if opt.lower() == 'adam':
        optimizer = optimizers.Adam(learning_rate=lr)
    elif opt.lower() == 'sgd':
        optimizer = optimizers.SGD(learning_rate=lr)
    elif opt.lower() == 'rmsprop':
        optimizer = optimizers.RMSprop(learning_rate=lr)
    elif opt.lower() == 'sgdw':
        sched = tf.keras.optimizers.schedules.CosineDecay(lr, 800)
        optimizer = tfa.optimizers.SGDW(learning_rate=sched, weight_decay=0.0005, momentum=0.9)
    else:
        print('optimizer not implemented')
        sys.exit(1)
    return optimizer


def get_model(arch, data_config,  weights=False, loss_type="cross-entropy", opt="adam", lr=1e-3, dropout_rate=0.2,
              ret_base_model=False, pool='avg'):
    """
    Creates a  model compiled on given loss function.
    :param arch: name of network architecture. For example, simple, ssdl, vgg16, and wrn-28-2.
    :param data_config: Details of dataset
    :param weights: use random or ImageNet pretrained weights. By default random
    :param loss_type: One of cross-entropy, triplet, arcface, or contrastive
    :param opt: Name of optimizer
    :param lr: learning rate
    :param dropout_rate: dropout rate used for VGG16, and WRn-28-2
    :param ret_base_model: return base model of bigger network models like VGG16 and WRN-28-2
    :return: tf.keras compiled model
    """
    if weights:
        weight = "imagenet"
    else:
        weight = None
    input_shape = (data_config.size, data_config.size, data_config.channels)
    # get base model
    conv_base, es = get_network(arch=arch, input_shape=input_shape, weights=weight, pool=pool)
    # add classification head
    model = models.Sequential()
    model.add(conv_base)
    if es > 0:  # extra layers are added for vgg and wrn
        model.add(layers.Dropout(dropout_rate, name="dropout_out"))
        model.add(layers.Dense(es, activation=None, name="fc1"))

    # set up loss
    metrics = []  # None
    if 'triplet' in loss_type:
        model.add(layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="l2-normalisation"))  # l2-normalisation
        loss = triplet_loss
        # import tensorflow_addons as tfa
        # loss = tfa.losses.TripletSemiHardLoss()

    else:   # default cross-entropy loss
        model.add(layers.Dense(data_config.nc, activation='softmax', name="fc_out"))
        loss = tf.keras.losses.SparseCategoricalCrossentropy()  # 'sparse_categorical_crossentropy'
        metrics = ['acc']

    learning_rate = lr
    optimizer = get_optimizer(opt, learning_rate)
    # compile model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    model.build((None,) + input_shape)
    # model.summary()
    if ret_base_model:
        return model, [conv_base, loss, optimizer, metrics]
    return model


def get_callbacks(verb):
    """
    Create tf.keras.callbacks
    :param verb: verbosity of the logs
    :return: callbacks list
    """
    calls = []
    if verb == 2:
        tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)
        calls.append(tqdm_callback)

    return calls

