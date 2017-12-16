import csv
import glob

import os
import random

from tqdm import tqdm

from siamese import siamese_network
import tensorflow as tf
import numpy as np
from time import gmtime, strftime
from data_input import input_data_label

import pickle

'''Trains Alexnet-based Siamese network on cat/dog images'''

# Given distances and labels (1 should be seperate, 0 should be close) finds the best threshold to seperate data.
def find_optimal_threshold(distances, labels, start=0, stop=2.0, step=0.01):
    accs = [np.mean((distances > threshold) == labels) for threshold in np.arange(start, stop, step)]
    best_acc = np.max(accs)
    best_threshold = np.argmax(accs) * 0.01
    return accs, best_acc, best_threshold

# Splits Dog/Cat training images into testing and training set.
# Only returns labels and filenames.
# returns [(train_filename1, label1), (train_filename2, label2), ...], [(test_filename1, label1), (test_filename2, label2), ...]
def get_split_data(percent_train=0.7):
    dog_images = glob.glob(os.path.join("train", "dog.*.jpg"))
    cat_images = glob.glob(os.path.join("train", "cat.*.jpg"))
    dog_labels = [0] * len(dog_images)
    cat_labels = [1] * len(cat_images)
    train_image_names = dog_images + cat_images
    train_image_labels = dog_labels + cat_labels

    data = list(zip(train_image_names, train_image_labels))
    random.shuffle(data)

    train_data = data[:int(percent_train * len(data))]
    test_data = data[int(percent_train * len(data)):]

    return train_data, test_data


#
def main():
    ### Parameters
    percent_train = 0.7
    feature_length = 10
    capacity = 2000
    batch_size = 128
    num_threads = 8

    #Currently due to network using all convolutions, only works correctly with 224 x 224 images which is origional alexnet input.
    height = 224
    width = 224
    channels = 3

    margin = 1.0
    learning_rate=1E-5
    dropout_keep_prob = 0.5

    epochs = 1000
    batches_per_epoch = 100


    # Record Directory
    time_string = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    store_directory = os.path.join('records',time_string)

    os.mkdir(store_directory)

    #Get data split into training and testing data.
    train_data, test_data = get_split_data(percent_train=percent_train)

    #Save the split information for future use.
    with open(os.path.join(store_directory, "train_data_names_labels.obj"), 'wb') as train_data_names_labels_file:
        pickle.dump(train_data, train_data_names_labels_file)

    with open(os.path.join(store_directory, "test_data_names_labels.obj"), 'wb') as test_data_names_labels:
        pickle.dump(test_data, test_data_names_labels)

    train_image_names, train_image_labels= zip(*train_data)
    test_image_names, test_image_labels= zip(*test_data)

    batch_images1, batch_images2, batch_labels, _, _ = input_data_label(train_image_names, train_image_labels, capacity, height, width, batch_size, num_threads)
    test_batch_images1, test_batch_images2, test_batch_labels, _, _ = input_data_label(test_image_names, test_image_labels, capacity, height, width, batch_size, num_threads)

    with siamese_network(height, width, channels, learning_rate=learning_rate, dropout_keep_prob=dropout_keep_prob, contrast_loss_margin=margin, feature_length=feature_length) as network:
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=network.sess, coord=coord)

        model_dir = os.path.join(store_directory, "model")
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        with open(os.path.join(store_directory, 'results.csv'), 'w', newline='\n') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(['iteration', 'average_train_loss', 'average_train_acc', 'average_test_loss', 'average_test_acc', 'best_training_acc',
                 'best_training_threshold', 'best_testing_acc', 'best_testing_threshold'])

            for e in tqdm(range(epochs), desc="Training progress"):
                t = tqdm(range(batches_per_epoch), desc="Epoch %i" % e, mininterval=0.5)

                #Training Step
                all_dist = []
                all_labels = []
                total_loss = 0
                for batch_counter in t:
                    images1, images2, labels = network.sess.run([batch_images1, batch_images2, batch_labels])
                    opt, l, dist = network.train(images1, images2, labels)

                    all_dist.extend(dist)
                    all_labels.extend(labels)

                    total_loss += l

                average_train_loss = total_loss/batches_per_epoch
                print("average_train_loss:", average_train_loss)

                training_acc, best_training_acc, best_training_threshold = find_optimal_threshold(all_dist, all_labels, 0, 2.0, 0.01)
                print('best_training_acc:', best_training_acc, ' best_training_threshold:',best_training_threshold)

                # Save session
                network.save(os.path.join(model_dir,'model'), global_step=e)

                total_test_loss = 0

                #Test Evaluation
                all_dist = []
                all_labels = []
                test_epochs = 10
                for test_batch in tqdm(range(test_epochs)):
                    images1, images2, labels = network.sess.run([test_batch_images1, test_batch_images2, test_batch_labels])
                    l, sq_dist, feat1, feat2 = network.test(images1,images2, labels=labels)
                    dist = np.sqrt(sq_dist)
                    all_dist.extend(dist)
                    all_labels.extend(labels)

                    total_test_loss += l

                average_test_loss = total_test_loss/test_epochs
                testing_acc, best_testing_acc, best_testing_threshold = find_optimal_threshold(all_dist, all_labels,
                                                                                                  0, 2.0, 0.01)

                print('best_testing_acc:', best_testing_acc, ' best_testing_threshold:',best_testing_threshold)
                spamwriter.writerow([e, average_train_loss, average_test_loss, best_training_acc, best_training_threshold, best_testing_acc, best_testing_threshold])
                csvfile.flush()

            print('Finished training!')

        coord.request_stop()




    print("hi")


if __name__ == "__main__":
    main()