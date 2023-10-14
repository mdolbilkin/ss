import tensorflow as tf
from matplotlib import pyplot as plt

def convert_to_grayscale(images, labels):
  return tf.image.rgb_to_grayscale(images), labels
def Create_tf_datasets():
    # batch_size = 32
    # img_height = 150
    # img_width = 80
    # train_ds = tf.keras.utils.image_dataset_from_directory(
    #                             'generated_image',
    #                             validation_split=0.2,
    #                             subset="training",
    #                             seed=123,
    #                             image_size=(img_height, img_width),
    #                             batch_size=batch_size,
    #                             label_mode='categorical')
    
    # val_ds = tf.keras.utils.image_dataset_from_directory(
    #                             'generated_image',
    #                             validation_split=0.2,
    #                             subset="validation",
    #                             seed=123,
    #                             image_size=(img_height, img_width),
    #                             batch_size=batch_size,
    #                             label_mode='categorical')

    # class_names = train_ds.class_names # type: ignore
    # print(class_names)

    # autotune = tf.data.experimental.AUTOTUNE
    # train_ds = train_ds.cache().shuffle(2400).prefetch(buffer_size = autotune) # type: ignore
    # val_ds = val_ds.cache().shuffle(600).prefetch(buffer_size = autotune) # type: ignore
    # normalization_layer = tf.keras.layers.Rescaling(1./255)
    # train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    # val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    # return train_ds, val_ds, len(class_names)
    
    
    batch_size = 32
    img_height = 1000
    img_width = 1000
    train_ds = tf.keras.utils.image_dataset_from_directory(
                                'SNOTS',
                                validation_split=0.2,
                                subset="training",
                                seed=123,
                                image_size=(img_height, img_width),
                                batch_size=batch_size,
                                label_mode='categorical')

    val_ds = tf.keras.utils.image_dataset_from_directory(
                                'SNOTS',
                                validation_split=0.2,
                                subset="validation",
                                seed=123,
                                image_size=(img_height, img_width),
                                batch_size=batch_size,
                                label_mode='categorical')

    class_names = train_ds.class_names # type: ignore
    print(class_names)

    autotune = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1600).prefetch(buffer_size = autotune) # type: ignore
    val_ds = val_ds.cache().shuffle(400).prefetch(buffer_size = autotune) # type: ignore
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    train_ds = train_ds.map(convert_to_grayscale)
    val_ds = val_ds.map(convert_to_grayscale)
    return train_ds, val_ds, len(class_names)

if __name__ == '__main__':
   pass