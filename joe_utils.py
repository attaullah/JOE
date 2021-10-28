import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, GlobalAvgPool2D, Flatten, Lambda, AvgPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
from models.model import get_callbacks, get_optimizer
from utils.shallow_classifiers import shallow_clf_accuracy
import time
import os
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


class JoeModel(tf.keras.Model):
    def __init__(self, size=32, channels=3, emb_size=64, num_classes=-1, loss_all_layers=True):
        super(JoeModel, self).__init__()
        self.num_classes = num_classes
        self.loss_all_layers = loss_all_layers
        self.conv1 = Conv2D(192, 5, input_shape=(size, size, channels), padding='same', activation='relu')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(160, 1, padding='same', activation='relu')
        self.bn2 = BatchNormalization()
        self.conv3 = Conv2D(96, 1, padding='same', activation='relu')
        self.bn3 = BatchNormalization()
        self.ap = AvgPool2D()
        # self.mp = MaxPool2D(pool_size=3, strides=2, padding='same', )
        self.conv4 = Conv2D(96, 5, padding='same', activation='relu')
        self.bn4 = BatchNormalization()
        self.conv5 = Conv2D(192, 1, padding='same', activation='relu')
        self.bn5 = BatchNormalization()
        self.conv6 = Conv2D(192, 1, padding='same', activation='relu')
        self.bn6 = BatchNormalization()
        self.conv7 = Conv2D(192, 3, padding='same', activation='relu')
        self.bn7 = BatchNormalization()
        self.conv8 = Conv2D(emb_size, 1, padding='same', activation='linear')
        self.bn8 = BatchNormalization()
        self.gap = GlobalAvgPool2D()

        self.l2_normalize = Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="embeddings")
        if self.num_classes > 0:
            self.fc = Dense(self.num_classes, name="last")

        self.flatten = Flatten()

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.loss1_tracker = tf.keras.metrics.Mean(name="loss1")
        self.loss2_tracker = tf.keras.metrics.Mean(name="loss2")
        self.loss3_tracker = tf.keras.metrics.Mean(name="loss3")
        self.loss4_tracker = tf.keras.metrics.Mean(name="loss4")
        self.loss5_tracker = tf.keras.metrics.Mean(name="loss5")
        self.loss6_tracker = tf.keras.metrics.Mean(name="loss6")
        self.loss7_tracker = tf.keras.metrics.Mean(name="loss7")
        self.loss8_tracker = tf.keras.metrics.Mean(name="loss8")
        self.acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.ap(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.ap(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.gap(x)
        if self.num_classes > 0:
            x = self.fc(x)
        else:
            x = self.l2_normalize(x)

        return x

    @tf.function
    def train_step(self, data, **kwargs):
        # Unpack the data.
        imgs, lbls = data
        # Forward pass
        with tf.GradientTape() as tape:
            x = self.conv1(imgs)
            x = self.bn1(x)
            loss1 = self.loss(lbls, self.l2_normalize(self.gap(x)))

            x = self.conv2(x)
            x = self.bn2(x)
            loss2 = self.loss(lbls, self.l2_normalize(self.gap(x)))

            x = self.conv3(x)
            x = self.bn3(x)
            x = self.ap(x)
            loss3 = self.loss(lbls, self.l2_normalize(self.gap(x)))

            x = self.conv4(x)
            x = self.bn4(x)
            loss4 = self.loss(lbls, self.l2_normalize(self.gap(x)))

            x = self.conv5(x)
            x = self.bn5(x)
            loss5 = self.loss(lbls, self.l2_normalize(self.gap(x)))

            x = self.conv6(x)
            x = self.bn6(x)
            x = self.ap(x)
            loss6 = self.loss(lbls, self.l2_normalize(self.gap(x)))

            x = self.conv7(x)
            x = self.bn7(x)
            loss7 = self.loss(lbls, self.l2_normalize(self.gap(x)))

            x = self.conv8(x)
            x = self.bn8(x)
            x = self.gap(x)

            if self.num_classes > 0:
                output = self.fc(x)
                lat_layer_params = (self.conv8.trainable_variables + self.bn8.trainable_variables +
                                    self.fc.trainable_variables)
            else:
                output = self.l2_normalize(x)
                lat_layer_params = (self.conv8.trainable_variables + self.bn8.trainable_variables)

            loss8 = self.loss(lbls, output)

            if self.loss_all_layers:
                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8
            else:
                loss = loss8

        # Compute gradients and update the parameters.
        learnable_params = (self.conv1.trainable_variables + self.bn1.trainable_variables +
                            self.conv2.trainable_variables + self.bn2.trainable_variables +
                            self.conv3.trainable_variables + self.bn3.trainable_variables +
                            self.conv4.trainable_variables + self.bn4.trainable_variables +
                            self.conv5.trainable_variables + self.bn5.trainable_variables +
                            self.conv6.trainable_variables + self.bn6.trainable_variables +
                            self.conv7.trainable_variables + self.bn7.trainable_variables +
                            lat_layer_params
                            )
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        # Monitor losses.
        self.loss_tracker.update_state(loss)
        self.loss1_tracker.update_state(loss1)
        self.loss2_tracker.update_state(loss2)
        self.loss3_tracker.update_state(loss3)
        self.loss4_tracker.update_state(loss4)
        self.loss5_tracker.update_state(loss5)
        self.loss6_tracker.update_state(loss6)
        self.loss7_tracker.update_state(loss7)
        self.loss8_tracker.update_state(loss8)
        self.acc_tracker.update_state(lbls, output)
        if self.num_classes > 0:
            return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}
        else:
            return {"total-loss": self.loss_tracker.result(), "loss1": self.loss1_tracker.result(),
                    "loss2": self.loss2_tracker.result(), "loss3": self.loss3_tracker.result(),
                    "loss4": self.loss4_tracker.result(), "loss5": self.loss5_tracker.result(),
                    "loss6": self.loss6_tracker.result(), "loss7": self.loss7_tracker.result(),
                    "loss8": self.loss8_tracker.result()
                    }

    def test_step(self, data, training=False):
        # Unpack the data.
        imgs, lbls = data
        x = self.conv1(imgs)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.ap(x)
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.conv5(x)
        x = self.bn5(x, training=training)
        x = self.conv6(x)
        x = self.bn6(x, training=training)
        x = self.ap(x)
        x = self.conv7(x)
        x = self.bn7(x, training=training)
        x = self.conv8(x)
        x = self.bn8(x, training=training)
        output = self.gap(x)
        if self.num_classes > 0:
            output = self.fc(output)

        loss = self.loss(lbls, output)
        self.acc_tracker.update_state(lbls, output)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}


def get_joe_model(data_config, opt='adam', lr=1e-3, loss_all_layers=False):
    optimizer = get_optimizer(opt, lr)
    loss = tfa.losses.TripletSemiHardLoss()
    model = JoeModel(data_config.size, data_config.channels, loss_all_layers=loss_all_layers)
    model.compile(optimizer, loss, )
    model.build((None, data_config.size, data_config.size, data_config.channels))
    print(model.summary())
    return model


def do_training(model, images, labels, test_images, test_labels, train_iter=10, batch_size=100, print_test_error=False,
                verb=1, hflip=True, vf=20, csv_path=None):
    if len(labels) == 1000 or len(labels) == 73257:   # hflip false for SVHN
        hflip = False
    calls = get_callbacks(verb)
    vf = train_iter // vf
    if csv_path:
        csv = tf.keras.callbacks.CSVLogger(csv_path + '.csv')
        calls.append(csv)
        knn = KnnEvaluator((images, labels), (test_images, test_labels), test_every=vf, name='knn', csv_path=csv_path,
                           batch_size=batch_size)
        calls.append(knn)
    aug = ImageDataGenerator(width_shift_range=0.125, height_shift_range=0.125, fill_mode='nearest',
                             horizontal_flip=hflip)

    test_aug = ImageDataGenerator()
    test_generator = test_aug.flow(test_images, test_labels, batch_size=batch_size)

    train_generator = aug.flow(images, labels, batch_size=batch_size)
    steps_per_epoch = int(len(labels) / batch_size)

    if print_test_error:
        history = model.fit(train_generator, epochs=train_iter,  verbose=verb,
                            steps_per_epoch=steps_per_epoch, validation_data=test_generator,
                            validation_freq=vf, callbacks=calls)
    else:
        history = model.fit(train_generator, epochs=train_iter, verbose=verb, steps_per_epoch=steps_per_epoch,
                            callbacks=calls)
    return history, csv_path


def start_training(model, dso, epochs=10, semi=True, bs=100, csv=True):
    if semi:  # for N-labelled
        images, labels = dso.train.labeled_ds.images, dso.train.labeled_ds.labels
    else:  # for All-labelled
        images, labels = dso.train.images, dso.train.labels
    csv_path = ''
    if csv:
        csv_path = "./csvs/{}/{}-{}-{}-{}".format("training", str(len(labels)), str(images.shape[1]),
                                                  time.strftime("%d-%m-%Y-%H%M%S"), os.uname()[1])

    history = do_training(model, images, labels, dso.test.images, dso.test.labels, epochs, bs, csv_path=csv_path)

    return history, csv_path


class KnnEvaluator(Callback):
    def __init__(self, lab_data, val_data, test_every=10,  csv_path='', name='knn', batch_size=128):

        super(KnnEvaluator, self).__init__()
        self.labels = val_data[1]
        self.images = val_data[0]
        self.x_lab = lab_data[0]
        self.y_lab = lab_data[1]
        self.test_every = test_every
        self.csv_path = csv_path
        self.name = name
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.test_every == 0:
            test_feat = self.model.predict(self.images, batch_size=self.batch_size, verbose=0)
            lab_feat = self.model.predict(self.x_lab, batch_size=self.batch_size)
            _, acc = shallow_clf_accuracy(lab_feat, self.y_lab, test_feat, self.labels, name=self.name)
            print('-KNN Test accuracy : {:.4f} \n'.format(acc))
            if self.csv_path:
                csv_file = open(self.csv_path+'-knn.csv', "a")
                csv_file.write(str(epoch + 1) + "," + str(acc)+"\n")
                csv_file.close()

