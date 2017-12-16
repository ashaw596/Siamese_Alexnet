import tensorflow as tf

# Normalizes tensorflow image.
# Converts 0-255 to -1 to 1 pixel values
def normalize_image(image):

    image = tf.cast(image, tf.float32) / 255.
    image = (image - 0.5) / 0.5
    return image


def input_data_label(image_names, labels, capacity, height, width, batch_size, num_threads, shuffle=True):

    # Read each JPEG file
    with tf.device('/cpu:0'):
        filename, label_queue = tf.train.slice_input_producer([image_names, labels],
                                           shuffle = shuffle)
        value = tf.read_file(filename)
        image = tf.image.decode_jpeg(value, channels=3)

        #Preprocessing
        image = tf.image.random_flip_left_right(image)
        image = normalize_image(image)

        # Resize
        image = tf.image.resize_images(image, (height, width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # image = tf.image.resize_image_with_crop_or_pad(image, height, width)

        # Using asynchronous queues
        img_batch, label_batch = tf.train.batch([image, label_queue],
                                           enqueue_many=False,
                                           batch_size=batch_size*2,
                                           num_threads=num_threads,
                                           capacity=capacity)

        image1, image2 =tf.split(img_batch, 2,axis =0)
        label_batch1, label_batch2 = tf.split(label_batch, 2, axis=0)

        labels = tf.abs(label_batch1 - label_batch2)

        return image1, image2, labels, label_batch1, label_batch2
