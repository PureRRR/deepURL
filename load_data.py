import scipy.io as sio
from sklearn.preprocessing import normalize
import numpy as np
import tensorflow as tf


def getData(data, num_modalities = 3):

    fea = {}
    gnd = data['label'].flatten()

    for i in range(0, num_modalities):
        modal = str(i + 1)
        x = data['mode_' + modal]
        #x = np.float32(x)
        #fea[i] - ndarray
        fea[modal] = x
        print(fea[modal].shape)
    print('Feature shape: {} '.format(len(fea)))

    """
    gnd:[...](1-40)
    fea:{'1':[[],...,[]],...}
    """
    randomNum = 888
    np.random.seed(randomNum)
    np.random.shuffle(gnd)
    for i in range(0, num_modalities):
        np.random.seed(randomNum)
        modality = str(i + 1)
        # fea[i] - ndarray
        np.random.shuffle(fea[modality])

    return gnd, fea


def get_num_classes(gnd):
    return max(gnd)


def load_data(datapath, num_modalities, TEST_BATCH_SIZE, BATCH_SIZE):
    # datapath = 'convert_data/ORL.mat'
    # num_modalities = 3
    print('Dataset: {0}, modalities: {1}'.format(datapath, num_modalities))
    data = sio.loadmat(datapath)
    print(data)

    gnd, fea = getData(data, num_modalities=num_modalities)
    num_data = gnd.shape[0]
    print('Label shape: {}'.format(gnd.shape))
    print('Label classes: {}'.format(get_num_classes(gnd)))

    normalized_fea_training = {}
    normalized_fea_testing = {}
    for i in range(0, num_modalities):
        modality = str(i + 1)
        n_f = normalize(fea[modality], axis=0, norm='max')
        n_f_training = n_f[0:num_data, :]
        n_f_testing = n_f[0:TEST_BATCH_SIZE, :]
        one_type_normalized_fea = {modality: n_f_training}
        one_type_normalized_fea_testing = {modality: n_f_testing}
        normalized_fea_training.update(one_type_normalized_fea)
        normalized_fea_testing.update(one_type_normalized_fea_testing)

    gnd_training = gnd[0:num_data]
    print(gnd_training)
    gnd_testing = gnd[0:TEST_BATCH_SIZE]
    normalized_fea_training.update({'label': gnd_training})
    dataset = tf.data.Dataset.from_tensor_slices(normalized_fea_training)
    normalized_fea_testing.update({'label': gnd_testing})
    test_dataset = tf.data.Dataset.from_tensor_slices(normalized_fea_testing)
    print('Normalized dataset: ')
    print(dataset)
    dataset = tf.data.Dataset.shuffle(dataset, len(dataset))
    dataset = dataset.batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.shuffle(test_dataset, len(test_dataset))
    test_dataset = test_dataset.batch(TEST_BATCH_SIZE)
    print('The number of batches contained in batched dataset: ' + str(len(dataset)))
    # print('The number of batches contained in labels: ' + str(len(labels)))

    return dataset, test_dataset, fea