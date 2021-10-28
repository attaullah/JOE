import os
from abc import ABC

import numpy as np
from sklearn.metrics import accuracy_score
from utils.utils import feature_scaling
from utils.shallow_classifiers import shallow_clf_accuracy
from scipy.spatial.distance import cdist
# suppress TF low level logging info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.model import get_model, get_callbacks
from data_utils import get_dataset
import tensorflow as tf
import time
from utils.ramps import linear_rampup
from data_utils.SS_generator import combined_pseudo_generator
template1 = "Labeled= {} selection={}% iterations= {}"
template2 = 'total selected based on percentile {} having accuracy {:.2f}%'
template3 = 'Length of predicted {},  unlabeled {}'


def set_dataset(dataset, lt, semi=True, scale=True):
    """
    Create dataset object containing labeled, unlabeled, and test datasets
    :param dataset: Name of dataset. mnist, fashion_mnist, svhn_cropped, cifar10, plant32, plant64 or plant96
    :param lt: One of cross-entropy, triplet, arcface, or contrastive
    :param semi: semi=true means N-labelled, semi=False means all-labelled
    :param scale: between 0...1
    :return: dataset object and dataset details
    """
    one_hot = True if lt.lower() == "arcface" else False
    dso, data_config = get_dataset.read_data_sets(dataset, one_hot, semi, scale=scale)
    return dso, data_config


def set_model(arch, data_config, weights, loss_type="", opt="adam", lr=1e-3):
    """
    setup a model compiled based on given loss function
    :rtype: tf.keras.model
    """
    model = get_model(arch, data_config, weights, loss_type, opt, lr)
    model.summary()
    return model


def get_log_name(flags, data_config, prefix=""):
    """
    Creates a name for target directory and file for saving training logs
    :param flags: training flags
    :param data_config: dataset details
    :param prefix: string to be added
    :return: path of directory and name of file for saving logs
    """
    path = prefix + flags.lt + '_logs/' + flags.dataset + '/' + flags.network + '/'

    log_name = str(data_config.n_label) + '-'
    weights = '-w' if flags.weights else ''
    if flags.lt != "cross-entropy":
        log_name += flags.lbl + '-'

    log_name += flags.opt.lower() + weights
    if flags.self_training:
        log_name = log_name + '-self-training-'
        if flags.lt != "cross-entropy":
            log_name += flags.confidence_measure

    return path, log_name


def compute_supervised_accuracy(model, imgs, lbls, ret_labels=False, v=0, bs=100):
    """
    Evaluates accuracy of the model from softmax probabilities.
    :param model: tf.keras.model
    :param imgs: test images
    :param lbls: test labels
    :param ret_labels: return predicted labels or not
    :param v: verbose. By default 0.
    :param bs: size of mini-batch
    :return: accuracy or accuracy and predicted labels
    """
    accuracy = model.evaluate(imgs, lbls, verbose=v, batch_size=bs)
    if isinstance(accuracy, list):
        accuracy = accuracy[1]
    accuracy = np.round(accuracy * 100., 2)
    if ret_labels:
        pred_lbls = model.predict(imgs, verbose=v, batch_size=bs)
        return accuracy, pred_lbls
    return accuracy


def get_network_embeddings(model, input_images, lt,  bs=100):
    """
    Returns network embeddings, can be used for accuracy calculation for metric learning losses.
    :param model: tf.keras.model
    :param input_images: input images
    :param lt: loss_type: One of cross-entropy, triplet, arcface, or contrastive
    :param bs: size of mini-batch
    :return: embeddings
    """
    return model.predict(input_images, batch_size=bs)
    # if 'face' in lt:
    #     inp = model.input[0]                        # input placeholder
    #     layer = -3
    # else:
    #     inp = model.input  # input placeholder
    #     layer = -2
    # outputs = model.layers[layer].output       # all layer outputs
    # emb = models.Model(inp, outputs)
    # feat = emb.predict(input_images, batch_size=bs)
    #
    # return feat


def get_network_output(model, input_images, lt="", scaling=False, v=0, bs=100):
    """
    Returns the output of network model on given input images.
    :param model: tf.keras.model
    :param input_images: input images
    :param lt: loss_type: One of cross-entropy, triplet, arcface, or contrastive
    :param scaling: perform feature scaling
    :param v: verbosity
    :param bs: size of mini-batch
    :return: output features
    """
    if 'cross-entropy' in lt:
        feat = model.predict(input_images, verbose=v, batch_size=bs)
    else:
        feat = get_network_embeddings(model, input_images, lt)
    if scaling:
        feat, _, _ = feature_scaling(feat)
    return feat


def compute_embeddings_accuracy(model, imgs, lbls, test_imgs, test_lbls, labelling="knn", loss_type="",
                                ret_labels=False, scaling=True):
    """
    Calculate accuracy from embeddings for metric learning losses.
    :param model: tf.keras.model
    :param imgs: labeled images
    :param lbls: labels of labeled images
    :param test_imgs: test images
    :param test_lbls: labels of test images
    :param labelling: type of shallow classifier. can be one of knn: k-nearest-neighbor, lda: linear discriminant
    analysis, rf: random forest, and lr: logistic regression
    :param loss_type: loss_type: One of cross-entropy, triplet, arcface, or contrastive
    :param ret_labels: whether to return predicted labels
    :param scaling: apply feature scaling or not
    :return: test accuracy
    """
    if lbls.ndim > 1:  # for Arcface, convert one-hot encodings to simple
        lbls = np.argmax(lbls, 1)
        test_lbls = np.argmax(test_lbls, 1)
    labels, accuracy = shallow_clf_accuracy(get_network_output(model, imgs, loss_type, scaling=scaling), lbls,
                                            get_network_output(model, test_imgs, loss_type, scaling=scaling),
                                            test_lbls, labelling)
    accuracy = np.round(accuracy * 100., 2)
    if ret_labels:
        return accuracy, labels
    return accuracy


def compute_accuracy(model, train_images, train_labels, test_images, test_labels, loss_type="cross-entropy",
                     labelling="knn"):
    """
    computes test accuracy from either softmax probabilities or from embeddings by training a shallow classifier.
    :rtype: accuracy
    """
    if 'cross-entropy' in loss_type:
        ac = compute_supervised_accuracy(model, test_images, test_labels)
    else:
        ac = compute_embeddings_accuracy(model, train_images, train_labels, test_images, test_labels,
                                         loss_type=loss_type, labelling=labelling)
    return ac


def log_accuracy(model, dso, loss_type="", semi=True, labelling="knn"):
    """
        computes test accuracy from dataset object either by using softmax probabilities or  embeddings by training a
        shallow classifier.

    :rtype: accuracy
    """
    if semi:
        acc = compute_accuracy(model, dso.train.labeled_ds.images, dso.train.labeled_ds.labels, dso.test.images,
                               dso.test.labels, loss_type=loss_type, labelling=labelling)
    else:
        acc = compute_accuracy(model, dso.train.images, dso.train.labels, dso.test.images, dso.test.labels,
                               loss_type=loss_type, labelling=labelling)
    return acc


def start_training(model, dso, epochs=100, semi=True, bs=100, verb=1):
    """
    Starts training
    :param model: tf.keras.model
    :param dso: dataset object containing train.labeled, train.unlabeled, and test datasets
    :param epochs: training for the number of epochs
    :param semi: semi=True : N-labelled, semi=False: All-labelled
    param verb:
    :param bs:
    """
    if semi:  # N-labelled
        images, labels = dso.train.labeled_ds.images, dso.train.labeled_ds.labels
    else:  # all-labelled examples
        images, labels = dso.train.images, dso.train.labels,

    do_training(model, images, labels, dso.test.images, dso.test.labels, train_iter=epochs, batch_size=bs, verb=verb)


def do_training(model, images, labels, test_images, test_labels, train_iter=10, batch_size=100, print_test_error=False,
                verb=1, hflip=True, vf=1, iter='', pseudo_labels=None, pseudo_images=None):

    calls = get_callbacks(verb)
    csv_path = "./csvs/{}-{}-supervised-{}-{}.csv".format(iter, str(len(labels)), time.strftime("%d-%m-%Y-%H%M%S"),
                                                          os.uname()[1])
    # print("saving logs at ", csv_path)
    csv = tf.keras.callbacks.CSVLogger(csv_path)
    # calls.append(csv)
    aug = ImageDataGenerator(width_shift_range=0.125, height_shift_range=0.125, fill_mode='nearest',
                             horizontal_flip=hflip)

    test_aug = ImageDataGenerator()
    test_generator = test_aug.flow(test_images, test_labels, batch_size=batch_size)

    if pseudo_images is not None:

        labeled_data = aug.flow(images, labels, shuffle=True, batch_size=1)
        pseudo_data = aug.flow(pseudo_images, pseudo_labels, shuffle=True, batch_size=1)
        train_generator = combined_pseudo_generator(labeled_data, pseudo_data, batch_size)
        steps_per_epoch = int(len(pseudo_labels) / batch_size)
    else:
        train_generator = aug.flow(images, labels, batch_size=batch_size)
        steps_per_epoch = int(len(labels) / batch_size)

    if print_test_error:
        history = model.fit(train_generator, epochs=train_iter,  verbose=verb,
                            steps_per_epoch=steps_per_epoch, validation_data=test_generator,
                            validation_freq=vf, callbacks=calls)
    else:
        history = model.fit(train_generator, epochs=train_iter, verbose=verb, steps_per_epoch=steps_per_epoch,
                            callbacks=calls)
    return history


class CustomModel(tf.keras.Model):
    def __init__(self, base_model, lt, emb_size=64, dropout_rate=0.2, num_classes=10):
        super(CustomModel, self).__init__()

        self.base_model = base_model
        self.classification_head = tf.keras.Sequential([
            tf.keras.layers.Dropout(dropout_rate, name="dropout_out"),
            tf.keras.layers.Dense(emb_size, name="embeddings")
        ])
        if 'triplet' in lt:
            self.classification_head.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1),
                                         name="l2-normalisation"))  # l2-normalisation

        else:  # default cross-entropy loss
            self.classification_head.add(tf.keras.layers.Dense(num_classes, name="fc_out"))

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")
        # self.val_loss_tracker = tf.keras.metrics.Mean(name="val-loss")
        # self.val_acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="val-acc")

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.classification_head(x)
        return x

    def train_step(self, data):
        # Unpack the data.
        imgs, lbls = data
        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            base_output = self.base_model(imgs)
            output = self.classification_head(base_output)
            loss = self.loss(lbls, output)

        # Compute gradients and update the parameters.
        learnable_params = (
                self.base_model.trainable_variables + self.classification_head.trainable_variables
        )
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(lbls, output)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    def test_step(self, data):
        # Unpack the data.
        imgs, lbls = data
        base_output = self.base_model(imgs)
        output = self.classification_head(base_output)
        loss = self.loss(lbls, output)
        self.acc_tracker.update_state(lbls, output)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}


class PseudoLabel(tf.keras.Model):
    def __init__(self, base_model, lt, emb_size=64, dropout_rate=0.2, num_classes=10, pseudo_wt=0.0, mode="supervised"):
        super(PseudoLabel, self).__init__()
        self.pseudo_wt = pseudo_wt
        self.mode = mode
        self.base_model = base_model
        self.classification_head = tf.keras.Sequential([
            tf.keras.layers.Dropout(dropout_rate, name="dropout_out"),
            tf.keras.layers.Dense(emb_size, name="embeddings")
        ])
        if 'triplet' in lt:
            self.classification_head.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1),
                                         name="l2-normalisation"))  # l2-normalisation

        else:  # default cross-entropy loss
            self.classification_head.add(tf.keras.layers.Dense(num_classes, name="fc_out"))

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")
        self.lab_loss_tracker = tf.keras.metrics.Mean(name="lab-loss")
        self.pseudo_loss_tracker = tf.keras.metrics.Mean(name="pseudo-loss")
        self.pseudo_acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="pseudo-acc")

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.classification_head(x)
        return x

    def combined_pseudo_step(self, data):
        labeled, pseudo = data
        labeled_imgs, labeled_lbls = labeled
        pseudo_imgs, pseudo_lbls = pseudo
        # print()
        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            labeled_output = self.base_model(labeled_imgs)
            labeled_output = self.classification_head(labeled_output)
            pseudo_output = self.base_model(pseudo_imgs)
            pseudo_output = self.classification_head(pseudo_output)

            labeled_loss = self.loss(labeled_lbls, labeled_output)
            pseudo_loss = self.loss(pseudo_lbls, pseudo_output)
            loss = labeled_loss + self.pseudo_wt * pseudo_loss

        # Compute gradients and update the parameters.
        learnable_params = (self.base_model.trainable_variables + self.classification_head.trainable_variables)
        gradients = tape.gradient(loss, learnable_params)
        # Monitor loss.
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(labeled_lbls, labeled_output)
        self.lab_loss_tracker.update_state(labeled_loss)
        self.pseudo_loss_tracker.update_state(pseudo_loss)
        self.pseudo_acc_tracker.update_state(pseudo_lbls, pseudo_output)

        return_str = {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result(), "lab-loss":
                      self.lab_loss_tracker.result(), "pseudo-loss": self.pseudo_loss_tracker.result(),
                      "pseudo-acc": self.pseudo_acc_tracker.result(), "pseudo-wt": self.pseudo_wt}
        return gradients, learnable_params, return_str

    def supervised_step(self, data):
        imgs, lbls = data
        with tf.GradientTape() as tape:
            output = self.base_model(imgs)
            output = self.classification_head(output)
            loss = self.loss(lbls, output)
        # Compute gradients
        learnable_params = (self.base_model.trainable_variables + self.classification_head.trainable_variables)
        gradients = tape.gradient(loss, learnable_params)
        # Monitor stats.
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(lbls, output)

        return_str = {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}
        return gradients, learnable_params, return_str

    def train_step(self, data):
        if self.mode == "pseudo":
            gradients, learnable_params, return_str = self.combined_pseudo_step(data)
        else:  # self.mode == "pretrain":
            gradients, learnable_params, return_str = self.supervised_step(data)

        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        return return_str

    def test_step(self, data):
        # Unpack the data.
        imgs, lbls = data
        base_output = self.base_model(imgs)
        output = self.classification_head(base_output)
        loss = self.loss(lbls, output)
        self.acc_tracker.update_state(lbls, output)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}


def get_custom_model(arch, data_config, weights, loss_type="", opt="adam", lr=1e-3):
    _, [conv_base, _, optimizer, _] = get_model(arch, data_config, weights, loss_type, opt, lr,
                                                ret_base_model=True)

    model = CustomModel(conv_base, loss_type, num_classes=data_config.nc)
    if loss_type == "triplet":
        # from losses.Triplet import triplet_loss
        import tensorflow_addons as tfa
        loss = tfa.losses.TripletSemiHardLoss()  # triplet_loss
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer, loss=loss, metrics=['acc', tf.keras.metrics.SparseCategoricalAccuracy()])
    input_shape = (None, data_config.size, data_config.size, data_config.channels)
    model.build(input_shape=input_shape)
    model.summary()
    return model


def get_pseudo_model(arch, data_config, weights, loss_type="", opt="adam", lr=1e-3):
    _, [conv_base, _, optimizer, _] = get_model(arch, data_config, weights, loss_type, opt, lr,
                                                ret_base_model=True)
    model = PseudoLabel(conv_base, loss_type, num_classes=data_config.nc)
    if loss_type == "triplet":
        # from losses.Triplet import triplet_loss
        import tensorflow_addons as tfa
        loss = tfa.losses.TripletSemiHardLoss()  # triplet_loss
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer, loss=loss, metrics=['acc', tf.keras.metrics.SparseCategoricalAccuracy()])
    input_shape = (None, data_config.size, data_config.size, data_config.channels)
    model.build(input_shape=input_shape)
    model.summary()
    return model


def do_training_tf_data(model, images, labels, test_images, test_labels, train_iter=10, batch_size=100,
                        print_test_error=False, verb=1, hflip=True, vf=10):

    calls = get_callbacks(verb)
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds_test = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)
    ds = ds.batch(batch_size)

    steps_per_epoch = int(len(labels) / batch_size)

    if print_test_error:
        history = model.fit(ds, epochs=train_iter,  verbose=verb,
                            steps_per_epoch=steps_per_epoch, validation_data=ds_test,
                            validation_freq=vf, callbacks=calls)
    else:
        history = model.fit(ds, epochs=train_iter, verbose=verb, steps_per_epoch=steps_per_epoch,
                            callbacks=calls)
    return history


def start_self_learning(model, dso, dc, lt, i, mti, bs, logger):

    self_learning(model, dso, lt, logger,  i, dc.sp, mti, bs)


def pseudo_label_selection(imgs, pred_lbls, scores, orig_lbls, p=0.05):
    """
    Select top p% pseudo-labels based on confidence score. Stratified selection.

    :param imgs: images
    :param pred_lbls: predicted labels
    :param scores: confidence score
    :param orig_lbls: original labels of images
    :param p: selection percentile p%
    :return: pseudo-labelled images and their indices
    """
    to_select = int(len(pred_lbls) * p)
    pseudo_images = []
    pseudo_labels = []
    orig_lbls_selected = []
    number_classes = np.unique(pred_lbls)
    per_class = to_select // len(number_classes)
    args = np.argsort(scores)
    indices = []
    for key in number_classes:  # for all classes
        selected = 0
        for index in args:
            if pred_lbls[index] == key:
                pseudo_images.append(imgs[index])
                pseudo_labels.append(pred_lbls[index])
                indices.append(index)
                orig_lbls_selected.append(orig_lbls[index])
                selected += 1
                if per_class == selected:
                    break
    orig_lbls_selected = np.array(orig_lbls_selected)
    pseudo_labels = np.array(pseudo_labels)
    if orig_lbls_selected.ndim > 1:
        acc = accuracy_score(np.argmax(orig_lbls_selected, 1), pseudo_labels) * 100.
    else:
        acc = accuracy_score(orig_lbls_selected, pseudo_labels) * 100.
    return np.array(pseudo_images), pseudo_labels, indices, acc


def assign_labels(model, train_labels, train_images, unlabeled_imgs, unlabeled_lbls, lt="cross-entropy"):
    """
    compute labels and prediction score. For cross-entropy loss, softmax probabilities are used for label assignment and
    prediction score. For metric learning losses, 1-nearest-neighbour or local learning with global consistency (LLGC)
    is used for label assignment and for prediction score.
    :param model: tf.keras.model
    :param train_labels: training labels
    :param train_images: training images
    :param unlabeled_imgs:
    :param unlabeled_lbls:
    :param lt: loss type
    :return: predicted labels, prediction score and accuracy for unlabelled examples
    """

    if unlabeled_lbls.ndim > 1:  # if labels are one-hot encoded
        train_labels = np.argmax(train_labels, 1)
        unlabeled_lbls = np.argmax(unlabeled_lbls, 1)

    if lt == "cross-entropy":
        test_image_feat = model.predict(unlabeled_imgs)
        pred_lbls = np.argmax(test_image_feat, 1)
        calc_score = np.max(test_image_feat, 1)
        calc_score = calc_score * -1.  # negate probs for same notion as distance
    else:   # for other loss functions
        # default to 1-NN distance as confidence score
        pred_lbls = []
        calc_score = []
        k = 1
        test_image_feat = get_network_output(model, unlabeled_imgs, lt)
        current_labeled_train_feat = get_network_output(model, train_images, lt)
        for j in range(len(test_image_feat)):
            search_feat = np.expand_dims(test_image_feat[j], 0)
            # calculate the sqeuclidean similarity and sort
            dist = cdist(current_labeled_train_feat, search_feat, 'sqeuclidean')
            rank = np.argsort(dist.ravel())
            pred_lbls.append(train_labels[rank[:k]])
            calc_score.append(dist[rank[0]])

    pred_lbls = np.array(pred_lbls)
    pred_lbls = pred_lbls.squeeze()
    pred_acc = accuracy_score(unlabeled_lbls, pred_lbls)*100.
    # print('predicted accuracy {:.2f} %'.format(pred_acc))
    calc_score = np.array(calc_score)
    pred_score = calc_score.squeeze()
    return pred_lbls, pred_score, pred_acc


def self_learning(model, mdso, lt,  logger, num_iterations=25, percentile=0.05, epochs=200, bs=100):
    """
    Apply self-learning

    :param model: tf.keras.model
    :param mdso: dataset object containing labeled, unlabeled, and test datasets
    :param lt: loss type
    :param logger: for printing/saving logs
    :param num_iterations: number of meta-iterations. Default 25
    :param percentile: selection percentile `p%` of pseudo-labels. Default `5%`
    :param epochs: number of epochs in each meta-iteration
    :param bs: mini-batch size
    :return: images [initially-labeled+pseudo-labelled], labels [initially-labeled+pseudo-labelled]
    """
    # Initial labeled data
    imgs = mdso.train.labeled_ds.images
    lbls = mdso.train.labeled_ds.labels
    # Initial unlabeled data
    unlabeled_imgs = mdso.train.unlabeled_ds.images
    unlabeled_lbls = mdso.train.unlabeled_ds.labels
    if lbls.ndim > 1:
        n_classes = len(np.unique(np.argmax(lbls, 1)))
    else:
        n_classes = len(np.unique(lbls))
    n_label = len(lbls)

    logger.info(template1.format(n_label, 100 * percentile, num_iterations))
    logger.info("i-th meta-iteration, unlabelled accuracy, pseudo-label accuracy,test accuracy")

    for i in range(num_iterations):
        print('=============== Meta-iteration = ', str(i + 1), '/', num_iterations, ' =======================')
        # 1- training
        do_training(model, imgs, lbls, mdso.test.images, mdso.test.labels, epochs, bs, iter=str(i+1))
        # 2- predict labels and confidence score
        pred_lbls, pred_score, unlabeled_acc = assign_labels(model, mdso.train.labeled_ds.labels,
                                                             mdso.train.labeled_ds.images, unlabeled_imgs,
                                                             unlabeled_lbls, lt)
        # 3- select top p% pseudo-labels
        pseudo_label_imgs, pseudo_labels, indices_of_selected, pseudo_labels_acc = \
            pseudo_label_selection(unlabeled_imgs, pred_lbls, pred_score, unlabeled_lbls, percentile)
        # 4- merging new labeled for next loop iteration
        imgs = np.concatenate([imgs, pseudo_label_imgs], axis=0)
        if lbls.ndim > 1:  # if one-hot encoded
            pseudo_labels = np.eye(n_classes)[pseudo_labels]
        lbls = np.concatenate([lbls, pseudo_labels], axis=0)
        # 5- remove selected pseudo-labelled data from unlabelled data
        unlabeled_imgs = np.delete(unlabeled_imgs, indices_of_selected, 0)
        unlabeled_lbls = np.delete(unlabeled_lbls, indices_of_selected, 0)

        #####################################################################################
        #  print/save accuracies and other information
        test_acc = compute_accuracy(model, mdso.train.labeled_ds.images, mdso.train.labeled_ds.labels, mdso.test.images,
                                    mdso.test.labels, lt)
        print(template2.format(len(indices_of_selected), pseudo_labels_acc))
        print(template3.format(len(lbls) - n_label, len(unlabeled_lbls)))
        print("Acc: unlabeled: {:.2f} %,  test  {:.2f} %".format(unlabeled_acc, test_acc))
        # ith meta-iteration, unlabelled accuracy, pseudo-label accuracy, test accuracy
        logger.info("{},{:.2f},{:.2f},{:.2f}".format(i + 1, unlabeled_acc, pseudo_labels_acc, test_acc))
        #####################################################################################

    return imgs, lbls


def self_learning_pseudo(model, mdso, lt,  logger, loss_fn, optimizer, num_iterations=200, percentile=0.05, epochs=1, bs=100):

    # Initial labeled data
    lab_imgs = mdso.train.labeled_ds.images
    lab_lbls = mdso.train.labeled_ds.labels
    # Initial unlabeled data
    unlabeled_imgs = mdso.train.unlabeled_ds.images
    unlabeled_lbls = mdso.train.unlabeled_ds.labels

    model.mode = "pseudo"
    logger.info(template1.format(len(lab_imgs), 100 * 0, num_iterations))
    logger.info("i-th meta-iteration,  pseudo-label accuracy,test accuracy")
    #
    aug = ImageDataGenerator(width_shift_range=0.125, height_shift_range=0.125, fill_mode='nearest',
                             horizontal_flip=True)
    from data_utils.SS_generator import combined_pseudo_generator
    labeled_data = aug.flow(lab_imgs, lab_lbls, shuffle=True, batch_size=1)
    steps_per_epoch = int(len(unlabeled_lbls) / bs)
    test_aug = ImageDataGenerator()
    test_generator = test_aug.flow(mdso.test.images, mdso.test.labels, batch_size=bs)
    calls = []
    verb = 1
    for i in range(num_iterations):
        print('epoch {}/{} '.format(str(i + 1), num_iterations))
        ramp_up = linear_rampup(i, num_iterations)
        # 1- predict labels
        pseudo_labels, _, pseudo_acc = assign_labels(model, lab_lbls, lab_imgs, unlabeled_imgs, unlabeled_lbls, lt)
        # 2- training
        pseudo_data = aug.flow(unlabeled_imgs, pseudo_labels, shuffle=True, batch_size=1)
        train_generator = combined_pseudo_generator(labeled_data, pseudo_data, bs)
        # history = model.fit(train_generator, epochs=epochs, verbose=verb, steps_per_epoch=steps_per_epoch,
        #                     callbacks=calls, validation_data=test_generator)
        pseudo_label_training(model, loss_fn, optimizer, train_generator, epochs=1, batch_size=bs, pseudo_wt=ramp_up,
                              steps_per_epoch=steps_per_epoch)
        #####################################################################################
        #  print/save accuracies and other information
        # test_acc =   # history.history['val_acc'][0]
        test_acc = compute_accuracy(model, lab_imgs, lab_lbls, mdso.test.images, mdso.test.labels, lt)

        # print("test acc from hist object ", history.history['val_acc'][0])
        # print("Acc: unlabeled: {:.2f} %,  test  {:.2f} %  rampup value {}".format(unlabeled_acc, test_acc, ramp_up))
        # ith meta-iteration, pseudo-label accuracy, test accuracy
        logger.info("{},{:.2f},{:.2f}".format(i + 1, pseudo_acc, test_acc))
        model.pseudo_wt = ramp_up  # update weight of pseudo-labels


@tf.function
def combined_pseudo_step(model, loss_fn, optimizer, x_labeled, y_labeled, x_pseudo, y_pseudo, pseudo_wt):

    with tf.GradientTape() as tape:
        labeled_output = model(x_labeled)
        pseudo_output = model(x_pseudo)

        labeled_loss = loss_fn(y_labeled, labeled_output)
        pseudo_loss = loss_fn(y_pseudo, pseudo_output)
        loss = labeled_loss + pseudo_wt * pseudo_loss

    # Compute gradients and update the parameters.
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    # Monitor loss.
    loss_tracker.update_state(loss)
    acc_tracker.update_state(labeled_lbls, labeled_output)
    lab_loss_tracker.update_state(labeled_loss)
    pseudo_loss_tracker.update_state(pseudo_loss)
    pseudo_acc_tracker.update_state(pseudo_lbls, pseudo_output)

    return_str = {"loss": loss_tracker.result(), "acc": acc_tracker.result(), "lab-loss":
                  lab_loss_tracker.result(), "pseudo-loss": pseudo_loss_tracker.result(),
                  "pseudo-acc": pseudo_acc_tracker.result(), "pseudo-wt": pseudo_wt}
    return return_str


def pseudo_label_training(model, loss_fn, optimizer, train_generator, epochs=1, batch_size=128, pseudo_wt=0.,
                          steps_per_epoch=1):
    for epoch in range(epochs):
        # print("\nStart of epoch %d" % (epoch,))
        train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        combined_loss = tf.keras.metrics.Mean()
        ul_loss = tf.keras.metrics.Mean()
        l_loss = tf.keras.metrics.Mean()
        pseudo_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        import pkbar
        kbar = pkbar.Kbar(target=steps_per_epoch, epoch=epoch, num_epochs=epochs, width=8, always_stateful=True)

        # Iterate over the batches of the dataset.
        for step, ([x_labeled, y_labeled], [x_pseudo, y_pseudo]) in enumerate(train_generator):
            # print(x_labeled.shape, y_labeled.shape, x_pseudo.shape, y_pseudo.shape)
            with tf.GradientTape() as tape:
                logits = model(x_labeled, training=True)  # Logits for this minibatch
                # Compute the loss value for this minibatch.
                loss_l = loss_fn(y_labeled, logits)
                logits_ul = model(x_pseudo, training=True)  # Logits for this ul minibatch
                # Compute the loss value for this minibatch.
                loss_ul = loss_fn(y_pseudo, logits_ul)
                loss_value = loss_l + pseudo_wt * loss_ul

            grads = tape.gradient(loss_value, model.trainable_weights)

            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            train_acc_metric.update_state(y_labeled, logits)
            pseudo_acc_metric.update_state(y_pseudo, logits_ul)
            combined_loss.update_state(loss_value)
            l_loss.update_state(loss_l)
            ul_loss.update_state(loss_ul)
            kbar.update(step, values=[("comb-loss", combined_loss.result()), ("l_acc", train_acc_metric.result()),
                                      ("l_loss", l_loss.result()), ("ul_acc", pseudo_acc_metric.result()),
                                      ("ul_loss", ul_loss.result()), ("pseudo_wt", pseudo_wt)])
            if step == steps_per_epoch:
                break
        # print("train acc sup {:.4f} pseudo acc {:.4f} combined loss {:.4f} l_loss {:.4f} ul_loss {:.4f} pseudo_wt {}".
        #       format(train_acc_metric.result(), pseudo_acc_metric.result(), combined_loss.result(), l_loss.result(),
        #              ul_loss.result(), pseudo_wt))
    return 0
