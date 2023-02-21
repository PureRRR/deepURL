import numpy
import numpy as np
import tensorflow as tf
from munkres import Munkres
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score, f1_score, \
    precision_score, recall_score


def best_map(L1, L2):
    # L1 should be the ground truth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)

    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        if c[i] >= nClass1:
            continue
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def purity_score(y_true, y_pred):
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    # Labels might be missing e.g with set like 0,2 where 1 is missing
    # First find the unique labels, then map the labels to an ordered set
    # 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bin
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def compute_and_print_scores(cluster_result, lbs, mode = 'train'):
    if len(cluster_result.shape)>1:
        cluster_result = (tf.argmax(cluster_result, 1) + 1).numpy()

    pred = np.array(cluster_result)
    gt = np.array(lbs)
    pred = np.array(best_map(gt, pred))
    err_nums = np.sum(gt != pred)
    acc = 1 - err_nums.astype(float) / (gt.shape[0])
    nmi = normalized_mutual_info_score(gt, pred)
    ari = adjusted_rand_score(gt, pred)
    f1 = f1_score(gt, pred, average='macro')
    precision = precision_score(gt, pred, average='macro')
    recall = recall_score(gt, pred, average='macro')
    purity = purity_score(gt, pred)
    print('This is valuation results:')
    out_string = f'ACC: {acc : .3f} NMI: {nmi : .3f} ARI: {ari : .3f} F1: {f1 : .3f} Precision: {precision : .3f} ' \
                 f'Recall: {recall : .3f} Purity: {purity : .3f}'
    print(out_string)
    tmp = out_string.split(' NMI:  ')
    tmp = tmp[0].split('ACC:  ')
    acc_tmp = float(tmp[1])

    if 'train' == mode:
        f = open('scores.txt', 'a+')
        f.write(out_string)
        f.write('\n')
        f.close()

    return acc_tmp
