import os, random

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


def split_data(data_path, train_size=0.75, val_size=0.15, test_size=0.1):
  '''
  Function splits images into 3 folders with train-val-test data
  ----------
  data_path - path to images files
  train_size - size of train data in percentage (range 0.0 - 1.0)
  val_size - size of val data in percentage (range 0.0 - 1.0)
  test_size - size of test data in percentage (range 0.0 - 1.0)
  ----------
  Function does not return any value
  '''

  assert 0.0 < train_size < 1.0 , 'Train data percentage should be given in range 0.0 - 1.0!'
  assert 0.0 < val_size < 1.0 , 'Val data percentage should be given in range 0.0 - 1.0!'
  assert 0.0 < test_size < 1.0 , 'Test data percentage should be given in range 0.0 - 1.0!'
  assert train_size + val_size + test_size == 1.0, 'train_size + val_size + test_size should be equal to 1.0!'

  # before splitting data into train-val-test creating folders to fill
  os.makedirs(os.path.join(data_path, 'train'), exist_ok=True)
  os.makedirs(os.path.join(data_path, 'val'), exist_ok=True)
  os.makedirs(os.path.join(data_path, 'test'), exist_ok=True)

  # reading names of files in folder
  names = os.listdir(data_path)
  # leaving only images' names
  names = [name for name in names if name.endswith('.jpg')]
  # shuffle images (just in case)
  random.shuffle(names)

  for idx_name, name in enumerate(names):
    # print(data_path)
    if idx_name < int(train_size * len(names)): # filling up the train
      os.replace(os.path.join(data_path, name), os.path.join(data_path, 'train', name))

    elif idx_name < int((train_size + val_size) * len(names)): # filling up the validation
      os.replace(os.path.join(data_path, name), os.path.join(data_path, 'val', name))

    else: # filling up the test
      os.replace(os.path.join(data_path, name), os.path.join(data_path, 'test', name))


def parse_image(img_path):
  '''
  Function to parse images and to make some convertations and preparations for furter use
  Might be used through mapping it on the whole dataset
  ----------
  img_path - path to the image
  ----------
  Function returns image
  '''

  image = tf.io.read_file(img_path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.uint8)

  return image


def plot_pie(pd_series):
  labels = pd_series.value_counts().index.tolist()
  counts = pd_series.value_counts().values.tolist()
  plt.pie(counts, labels=labels)
  plt.title('Pie chart for ' + pd_series.name)