import os
import pickle

from tqdm import tqdm

from data_input import input_data_label
from siamese import siamese_network
import tensorflow as tf
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

""" Visualizes Model with PCA of training and testing features"""

def plot(feats, labels):
    fig = plt.figure()

    c_list = ['r' if lab==1 else 'b' for lab in labels]
    x = feats[:, 0]
    y = feats[:, 1]

    plt.scatter(x, y, s=2, c=c_list)
    plt.show()

def main():
    store_directory = os.path.join('records', 'best')
    model_dir = os.path.join(store_directory, 'model')

    height = 224
    width = 224
    channels = 3
    learning_rate = 0
    dropout_keep_prob = 1.0
    margin = 1
    feature_length = 10
    capacity = 2000
    batch_size = 100
    num_threads = 8


    with siamese_network(height, width, channels, learning_rate=learning_rate, dropout_keep_prob=dropout_keep_prob,
                         contrast_loss_margin=margin, feature_length=feature_length) as network:
        network.load(tf.train.latest_checkpoint(os.path.join(model_dir)))

        # Save the split information for future use.
        with open(os.path.join(store_directory, "train_data_names_labels.obj"), 'rb') as train_data_names_labels_file:
            train_data = pickle.load(train_data_names_labels_file)

        with open(os.path.join(store_directory, "test_data_names_labels.obj"), 'rb') as test_data_names_labels:
            test_data = pickle.load(test_data_names_labels)

        train_image_names, train_image_labels = zip(*train_data)
        test_image_names, test_image_labels = zip(*test_data)

        train_batch_images1, train_batch_images2, batch_labels, train_batch_label1, train_batch_label2 = input_data_label(train_image_names, train_image_labels, capacity,
                                                                      height, width, batch_size, num_threads, shuffle=False)
        test_batch_images1, test_batch_images2, test_batch_labels, test_batch_label1, test_batch_label2 = input_data_label(test_image_names,
                                                                                     test_image_labels, capacity,
                                                                                     height, width, batch_size,
                                                                                     num_threads, shuffle=False)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=network.sess, coord=coord)

        train_feats = []
        train_labels = []
        for i in tqdm(range(len(train_image_names)//batch_size//2)):
            images1, images2, labels, label1, label2 = network.sess.run([train_batch_images1, train_batch_images2, batch_labels, train_batch_label1, train_batch_label2])
            feat1 = network.inference(images1)
            feat2 = network.inference(images2)
            train_feats.extend(feat1.tolist())
            train_feats.extend(feat2.tolist())
            train_labels.extend(label1.tolist())
            train_labels.extend(label2.tolist())
        train_feats = np.array(train_feats)

        all_dist = []
        all_labels = []
        test_feats = []
        test_labels = []
        for i in tqdm(range(len(test_image_labels) // batch_size//2)):
            images1, images2, labels, label1, label2 = network.sess.run([test_batch_images1, test_batch_images2, test_batch_labels, test_batch_label1, test_batch_label2])
            l, sq_dist, feat1, feat2 = network.test(images1,images2, labels=labels)
            test_feats.extend(feat1)
            test_feats.extend(feat2)
            test_labels.extend(label1)
            test_labels.extend(label2)

            dist = np.sqrt(sq_dist)
            all_dist.extend(dist)
            all_labels.extend(labels)

        test_feats = np.array(test_feats)

        accs = [np.mean((all_dist > threshold) == all_labels) for threshold in np.arange(0, 2.0, 0.01)]
        print(accs)


        pca = PCA(n_components=2)
        pca_train_feat = pca.fit_transform(train_feats)
        pca_test_feat = pca.transform(test_feats)

        plot(pca_train_feat, train_labels)
        plot(pca_test_feat, test_labels)







if __name__ == "__main__":
    main()