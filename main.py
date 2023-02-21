import glob
import time

import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os

from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.models import Model
from tqdm import tqdm

# from SpectralClustering import testSpec
from load_data import load_data
from metrics import compute_and_print_scores

print(tf.__version__)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def encoder(input_dimension, output_dimension):
    model = keras.Sequential()
    model.add(layers.Dense(512, input_shape=(input_dimension,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # model.add(layers.Dense(128))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    model.add(layers.Dense(128))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(output_dimension))
    model.add(layers.BatchNormalization())
    model.add(layers.Softmax())

    return model


def decoder(input_dimension, output_dimension):
    model = keras.Sequential()
    model.add(layers.Dense(128, input_shape=(input_dimension,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # model.add(layers.Dense(128))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    model.add(layers.Dense(512))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(output_dimension))
    model.add(layers.BatchNormalization())

    return model



def InductionModule(out_dimension, input_dimension=128):
    """
    the input of this module is the concatenated features h from{h1,h2...hn}
    and the output is a V dimensional weight vector w
    """
    model = keras.Sequential()

    model.add(layers.Dense(128, input_shape=(input_dimension,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # model.add(layers.Dense(128))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    model.add(layers.Dense(out_dimension))
    model.add(layers.BatchNormalization())

    model.add(Activation('sigmoid'))
    model.add(layers.Softmax())

    return model


def make_model(model_path, num_modalities):
    encoders = []
    decoders = []
    decodersZ = []
    dim_sum = 0

    model_path_att = model_path + 'attention.h5'
    for i in range(0, num_modalities):
        model_path_encoder = model_path + 'encoder' + str(i) + '.h5'
        model_path_decoder = model_path + 'decoder' + str(i) + '.h5'
        model_path_decoderZ = model_path + 'decoderZ' + str(i) + '.h5'
        if not os.path.exists(model_path_encoder):
            feature_dimension_i = fea[str(i + 1)].shape[1]
            print(feature_dimension_i)
            # dim_sum += feature_dimension_i
            encoders.append(encoder(input_dimension=feature_dimension_i, output_dimension=LATENT_DIM))
            decoders.append(decoder(input_dimension=LATENT_DIM, output_dimension=feature_dimension_i))
            decodersZ.append(decoder(input_dimension=LATENT_DIM, output_dimension=feature_dimension_i))
        else:
            encoders.append(keras.models.load_model(model_path_encoder))
            decoders.append(keras.models.load_model(model_path_decoder))
            decodersZ.append(keras.models.load_model(model_path_decoderZ))
    if os.path.exists(model_path_att):
        attention = keras.models.load_model(model_path_att)
    else:
        attention = AttentionModule(out_dimension=LATENT_DIM, input_dimension=num_modalities * LATENT_DIM)

    return encoders, decodersZ, decoders, attention


def save_model(model_path):
    for i in range(num_modalities):
        encoders[i].save(model_path + 'encoder' + str(i) + '.h5')
        decoders[i].save(model_path + 'decoder' + str(i) + '.h5')
        decodersZ[i].save(model_path + 'decoderZ' + str(i) + '.h5')
    attention.save(model_path + 'attention.h5')



def train_step(dataset_batch):
    with tf.GradientTape(persistent=True) as en_tape1, tf.GradientTape(persistent=True) as en_tape2, tf.GradientTape(persistent=True) as en_tape3, \
            tf.GradientTape(persistent=True) as deZ_tape1, tf.GradientTape(persistent=True) as deZ_tape2, tf.GradientTape(persistent=True) as deZ_tape3, \
            tf.GradientTape() as de_tape1, tf.GradientTape() as de_tape2, tf.GradientTape() as de_tape3, \
             tf.GradientTape(persistent=True) as att_tape:
        input_feature = []
        for j in range(num_modalities):
            input_feature.append(dataset_batch[str(j + 1)])
            input_feature[j] = tf.cast(input_feature[j], tf.float32)

        encoder_out = []
        decoder_out = []
        decoder_outZ = []
        for j in range(0, num_modalities):
            # n dz
            en_out = encoders[j](input_feature[j], training=True)
            encoder_out.append(en_out)
            decoder_out.append(decoders[j](en_out))
            if j == 0:
                concated_out = en_out
            else:
                concated_out = tf.concat([concated_out, en_out], 1)

        fused_representation = attention(concated_out)


        for j in range(0, num_modalities):
            decoder_outZ.append(decodersZ[j](fused_representation))
        autoencoderZ_loss = []
        # autoencoder_loss = []
        for j in range(0, num_modalities):
            autoencoderZ_loss.append(tf.norm(input_feature[j] - decoder_outZ[j]))
            # autoencoder_loss.append(tf.norm(input_feature[j] - decoder_out[j]))

        concated_out = decoder_outZ[0]
        for j in range(1, num_modalities):
            concated_out = tf.concat([concated_out, decoder_outZ[j]], 1)
        # commonZ = tf.reduce_mean(encoder_out, 0)
        Zc = fused_representation

        # n dz
        view1 = encoder_out[0]
        view2 = encoder_out[1]
        backz = Zc
        n, dz = encoder_out[0].shape

        # n dz dz
        pv1v2 = tf.matmul(tf.expand_dims(view1, 2), tf.expand_dims(view2, 1))
        pv1v2 = tf.reshape(pv1v2, [n, 1, dz, dz])
        # pv1v2 = tf.repeat(pv1v2, dz, 1)

        backz = tf.reshape(backz, [n, dz, 1, 1])
        # backz = tf.repeat(backz, dz, 2)
        # backz = tf.repeat(backz, dz, 3)

        # dz dz dz
        pzv1v2 = tf.reduce_mean(pv1v2 * backz, 0)


        # for j in range(n):
        #     if(j==0):
        #         p = tf.multiply(pv1v2[j], backz[j])
        #     else:
        #         p = p + tf.multiply(pv1v2[j], backz[j])
        #
        # # dz dz dz
        # pzv1v2 = p / n

        del backz, pv1v2, view1, view2

        assert pzv1v2.shape == (dz, dz, dz)

        # dz dz
        pzv2 = tf.reduce_sum(pzv1v2, 1)
        pzv1 = tf.reduce_sum(pzv1v2, 2)

        # dz
        pz = tf.reduce_sum(pzv2, 1)
        pz = tf.reshape(pz, [dz, 1, 1])
        pz = tf.repeat(pz, dz, 1)
        pz = tf.repeat(pz, dz, 2)

        assert pz.shape == (dz, dz, dz)

        # dz dz dz
        pzv2 = tf.reshape(pzv2, [dz, 1, dz])
        pzv2 = tf.repeat(pzv2, dz, 1)
        pzv1 = tf.reshape(pzv1, [dz, dz, 1])
        pzv1 = tf.repeat(pzv1, dz, 2)

        assert pzv2.shape == (dz, dz, dz)
        assert pzv1.shape == (dz, dz, dz)

        cmi = - 150 * tf.reduce_sum(pzv1v2 * tf.math.log(pzv1v2 * pz / (pzv2 * pzv1+1e-10) +1e-10))
        del pzv1v2, pz, pzv1, pzv2

        estimator = KMeans(n_clusters=num_classes)
        estimator.fit(Zc)
        label_pred = estimator.labels_

        en_tapes = [en_tape1, en_tape2, en_tape3]
        deZ_tapes = [deZ_tape1, deZ_tape2, deZ_tape3]
        de_tapes = [de_tape1, de_tape2, de_tape3]

        for j in range(num_modalities):
            grad_encoder_Z = en_tapes[j].gradient(autoencoderZ_loss[j], encoders[j].trainable_variables)
            # grad_encoder_de = en_tapes[j].gradient(0. * autoencoder_loss[j], encoders[j].trainable_variables)
            grad_mutual_en = en_tapes[j].gradient(cmi, encoders[j].trainable_variables)

            grad_decoder_Z = deZ_tapes[j].gradient(autoencoderZ_loss[j], decodersZ[j].trainable_variables)

            # grad_decoder = de_tapes[j].gradient(autoencoder_loss[j], decoders[j].trainable_variables)

            grad_att_cmi = att_tape.gradient(cmi, attention.trainable_variables)

            autoencoder_optimizer.apply_gradients((zip(grad_encoder_Z + grad_mutual_en, encoders[j].trainable_variables)))
            autoencoder_optimizer.apply_gradients((zip(grad_decoder_Z, decodersZ[j].trainable_variables)))
            # autoencoder_optimizer.apply_gradients(zip(grad_decoder, decoders[j].trainable_variables))

            att_optimizer.apply_gradients(zip(grad_att_cmi, attention.trainable_variables))


        total_loss = np.sum(autoencoderZ_loss) + cmi
        return total_loss.numpy(), np.sum(autoencoderZ_loss), cmi.numpy(), label_pred



def test(test_dataset):
    input_feature = []

    for batch in test_dataset:
        test_dataset = batch

    for j in range(num_modalities):
        input_feature.append(test_dataset[str(j + 1)])
        input_feature[j] = tf.cast(input_feature[j], tf.float32)

    encoder_out = []
    for j in range(0, num_modalities):
        # n dz
        en_out = encoders[j](input_feature[j], training=True)
        encoder_out.append(en_out)
    commonZ = tf.reduce_sum(encoder_out, 0)

    estimator = KMeans(n_clusters=num_classes)
    estimator.fit(commonZ)
    label_pred = estimator.labels_
    # tsne_visual(hidden_cluster_repre, lbs, 'tsne')
    # tsne_visual(generator_out[1], lbs, 'tsne1')

    sco = compute_and_print_scores(label_pred, test_dataset['label'].numpy().tolist(), mode='test')

    f_test = open('tst_sco.txt', 'a+')
    f_test.write(str(sco))
    f_test.write('\n')

    return sco


def train(dataset, epochs):
    global now_highest
    for epoch in range(epochs):

        with tqdm(total=len(dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='it', colour='white') as pbar:
            for dataset_batch in dataset:
                whole_loss, autoencoder_loss, cmi_loss, cluster_result = train_step(dataset_batch)
                pbar.set_postfix({'batch_loss': whole_loss, 'cmi_loss': cmi_loss})
                out = open("out.txt", "a+")
                out.write('batch_loss : ' + str(whole_loss) + '  auto : ' + str(autoencoder_loss) + '  CMI : ' + str(cmi_loss) + '\n')
                out.close()

                sco = compute_and_print_scores(cluster_result, dataset_batch['label'].numpy().tolist(), mode='train')
                sco_test = test(test_dataset)

                pbar.update(1)
                save_model(model_path)

                # if sco>=0.7:
                #     acctmp = test(test_dataset)
                if sco_test>now_highest:
                    save_model('./best_weights/')
                    now_highest = sco_test

        dataset = tf.data.Dataset.shuffle(dataset, len(dataset))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    num_modalities = 2
    LATENT_DIM = 64
    now_highest = 0.0

    autoencoder_optimizer = tf.keras.optimizers.Adam(1e-3)
    att_optimizer = tf.keras.optimizers.Adam(1e-4)


    EPOCH = 15
    datapath = 'convert_data/MNIST_2views.mat'
    num_classes = 20
    TEST_BATCH_SIZE = 2386
    BATCH_SIZE = 200
    LAMB_ATT = 1
    dataset, test_dataset, fea = load_data(datapath, num_modalities, TEST_BATCH_SIZE, BATCH_SIZE)

    model_path = './save_weights/'
    encoders, decodersZ, decoders, attention = make_model(model_path, num_modalities)

    # print('My model:')
    # test(test_dataset)
    # print('Compared model:')
    # testSpec(test_dataset, num_classes, num_modalities)
    for i_epo in range(25):
        train(dataset, EPOCH)
        # acctmp = test(test_dataset)
        # if acctmp>now_highest:
        #     save_model('./best_weights/now_highest/')
        #     now_highest = acctmp
    # test(test_dataset)
