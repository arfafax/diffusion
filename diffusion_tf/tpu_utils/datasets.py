"""Dataset loading utilities.

All images are scaled to [0, 255] instead of [0, 1]
"""

import functools

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def pack(image, label):
  label = tf.cast(label, tf.int32)
  return {'image': image, 'label': label}


class SimpleDataset:
  DATASET_NAMES = ('cifar10', 'celebahq256')

  def __init__(self, name, tfds_data_dir):
    self._name = name
    self._data_dir = tfds_data_dir
    self._img_size = {'cifar10': 32, 'celebahq256': 256}[name]
    self._img_shape = [self._img_size, self._img_size, 3]
    self._tfds_name = {
      'cifar10': 'cifar10:3.0.0',
      'celebahq256': 'celeb_a_hq/256:2.0.0',
    }[name]
    self.num_train_examples, self.num_eval_examples = {
      'cifar10': (50000, 10000),
      'celebahq256': (30000, 0),
    }[name]
    self.num_classes = 1  # unconditional
    self.eval_split_name = {
      'cifar10': 'test',
      'celebahq256': None,
    }[name]

  @property
  def image_shape(self):
    """Returns a tuple with the image shape."""
    return tuple(self._img_shape)

  def _proc_and_batch(self, ds, batch_size):
    def _process_data(x_):
      img_ = tf.cast(x_['image'], tf.int32)
      img_.set_shape(self._img_shape)
      return pack(image=img_, label=tf.constant(0, dtype=tf.int32))

    ds = ds.map(_process_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

  def train_input_fn(self, params):
    ds = tfds.load(self._tfds_name, split='train', shuffle_files=True, data_dir=self._data_dir)
    ds = ds.repeat()
    ds = ds.shuffle(50000)
    return self._proc_and_batch(ds, params['batch_size'])

  def train_one_pass_input_fn(self, params):
    ds = tfds.load(self._tfds_name, split='train', shuffle_files=False, data_dir=self._data_dir)
    return self._proc_and_batch(ds, params['batch_size'])

  def eval_input_fn(self, params):
    if self.eval_split_name is None:
      return None
    ds = tfds.load(self._tfds_name, split=self.eval_split_name, shuffle_files=False, data_dir=self._data_dir)
    return self._proc_and_batch(ds, params['batch_size'])


class LsunDataset:
  def __init__(self,
    tfr_file,            # Path to tfrecord file.
    resolution=256,      # Dataset resolution.
    max_images=None,     # Maximum number of images to use, None = use all images.
    shuffle_mb=4096,     # Shuffle data within specified window (megabytes), 0 = disable shuffling.
    buffer_mb=256,       # Read buffer size (megabytes).
  ):
    """Adapted from https://github.com/NVlabs/stylegan2/blob/master/training/dataset.py.
    Use StyleGAN2 dataset_tool.py to generate tf record files.
    """
    self.tfr_file           = tfr_file
    self.dtype              = 'int32'
    self.max_images         = max_images
    self.buffer_mb          = buffer_mb
    self.num_classes        = 1         # unconditional

    # Determine shape and resolution.
    self.resolution = resolution
    self.resolution_log2 = int(np.log2(self.resolution))
    self.image_shape = [self.resolution, self.resolution, 3]

  def _train_input_fn(self, params, one_pass: bool):
    # Build TF expressions.
    dset = tf.data.TFRecordDataset(self.tfr_file, compression_type='', buffer_size=self.buffer_mb<<20)
    if self.max_images is not None:
      dset = dset.take(self.max_images)
    if not one_pass:
      dset = dset.repeat()
    dset = dset.map(self._parse_tfrecord_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Shuffle and prefetch
    dset = dset.shuffle(50000)
    dset = dset.batch(params['batch_size'], drop_remainder=True)
    dset = dset.prefetch(tf.data.experimental.AUTOTUNE)
    return dset

  def train_input_fn(self, params):
    return self._train_input_fn(params, one_pass=False)

  def train_one_pass_input_fn(self, params):
    return self._train_input_fn(params, one_pass=True)

  def eval_input_fn(self, params):
    return None

  # Parse individual image from a tfrecords file into TensorFlow expression.
  def _parse_tfrecord_tf(self, record):
    features = tf.parse_single_example(record, features={
      'shape': tf.FixedLenFeature([3], tf.int64),
      'data': tf.FixedLenFeature([], tf.string)})
    data = tf.decode_raw(features['data'], tf.uint8)
    data = tf.cast(data, tf.int32)
    data = tf.reshape(data, features['shape'])
    data = tf.transpose(data, [1, 2, 0])  # CHW -> HWC
    data.set_shape(self.image_shape)
    return pack(image=data, label=tf.constant(0, dtype=tf.int32))

def dataset_parser_static(record):
    """Parses an image and its label from a serialized ResNet-50 TFExample.

        This only decodes the image, which is prepared for caching.

    Args:
        value: serialized string containing an ImageNet TFExample.

    Returns:
        Returns a tuple of (image, label) from the TFExample.
    """
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, ''),
        'image/format': tf.FixedLenFeature((), tf.string, 'jpeg'),
        'image/class/label': tf.FixedLenFeature([], tf.int64, -1),
        'image/class/embedding': tf.VarLenFeature(tf.float32),
        'image/width': tf.FixedLenFeature([], tf.int64, -1),
        'image/height': tf.FixedLenFeature([], tf.int64, -1),
        'image/filename': tf.FixedLenFeature([], tf.string, ''),
        'image/class/text': tf.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
    }

    parsed = tf.parse_single_example(record, keys_to_features)
    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])
    image = tf.io.decode_image(image_bytes, 3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize_image_with_pad(image, target_height=256, target_width=256, method=tf.image.ResizeMethod.AREA)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    data = tf.cast(image, tf.int32)
    data.set_shape([256, 256, 3])
    return pack(image=data, label=tf.constant(0, dtype=tf.int32))


class TForkDataset:
  def __init__(self,
    tfr_file,            # Path to tfrecord file.
    resolution=256,      # Dataset resolution.
    max_images=None,     # Maximum number of images to use, None = use all images.
    shuffle_mb=4096,     # Shuffle data within specified window (megabytes), 0 = disable shuffling.
    buffer_mb=256,       # Read buffer size (megabytes).
  ):
    self.tfr_file           = tfr_file
    self.dtype              = 'int32'
    self.max_images         = max_images
    self.buffer_mb          = buffer_mb
    self.num_classes        = 1         # unconditional

    # Determine shape and resolution.
    self.resolution = resolution
    self.resolution_log2 = int(np.log2(self.resolution))
    self.image_shape = [self.resolution, self.resolution, 3]

  def _train_input_fn(self, params, one_pass: bool):
      dset = self._make_dataset(
        data_dirs=self.tfr_file,
        index=TForkDataset._get_current_host(params),
        num_hosts=TForkDataset._get_num_hosts(params),
        buffer_mb=self._buffer_mb,
      )

      # cache the unparsed image data.
      dset = dset.cache()
      # fused shuffle and repeat.
      dset = dset.apply(tf.contrib.data.shuffle_and_repeat(1024 * 16))
      # parse the image data.
      assert "batch_size" in params
      dset = dset.apply(
        tf.contrib.data.map_and_batch(
          dataset_parser_static,
          batch_size=params["batch_size"],
          num_parallel_batches=TForkDataset._get_num_cores(params),
          drop_remainder=True))
      # prefetch the dataset.
      dset = dset.prefetch(tf.data.experimental.AUTOTUNE)
      return dset

  @staticmethod
  def _get_current_host(self, params):
    # TODO(dehao): Replace the following with params['context'].current_host
    if 'context' in params:
      return params['context'].current_input_fn_deployment()[1]
    elif 'dataset_index' in params:
      return params['dataset_index']
    else:
      return 0

  @staticmethod
  def _get_num_hosts(self, params):
    if 'context' in params:
     return params['context'].num_hosts
    elif 'dataset_index' in params:
      return params['dataset_num_shards']
    else:
      return 1

  @staticmethod
  def _get_num_cores(self, params):
    return 8 * self._get_num_hosts(params)



  @staticmethod
  def _make_dataset(data_dirs, index=0, num_hosts=1,
                   seed=None, shuffle_filenames=False,
                   num_parallel_calls = 64,
                   filename_shuffle_buffer_size = 100000,
                   buffer_mb = 256):

    if shuffle_filenames:
      assert seed is not None

    file_patterns = [x.strip() for x in data_dirs.split(',') if len(x.strip()) > 0]
    print(file_patterns)

    # For multi-host training, we want each hosts to always process the same
    # subset of files.  Each host only sees a subset of the entire dataset,
    # allowing us to cache larger datasets in memory.
    dataset = None
    for pattern in file_patterns:
      x = tf.data.Dataset.list_files(pattern, shuffle=shuffle_filenames, seed=seed)
      dataset = x if dataset is None else dataset.concatenate(x)
    dataset = dataset.shard(num_hosts, index)
    print(dataset)

    # Memoize the filename list to avoid lots of calls to list_files.
    dataset = dataset.cache()

    # For mixing multiple datasets, shuffle list of filenames.
    dataset = dataset.shuffle(filename_shuffle_buffer_size, seed=seed)

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024  # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_mb<<20)
      return dataset

    # Read the data from disk in parallel
    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            fetch_dataset, cycle_length=num_parallel_calls, sloppy=True))

    return dataset


  def train_input_fn(self, params):
    return self._train_input_fn(params, one_pass=False)

  def train_one_pass_input_fn(self, params):
    return self._train_input_fn(params, one_pass=True)

  def eval_input_fn(self, params):
    return None





DATASETS = {
  "cifar10": functools.partial(SimpleDataset, name="cifar10"),
  "celebahq256": functools.partial(SimpleDataset, name="celebahq256"),
  "lsun": LsunDataset,
  "tfork": TForkDataset,
}


def get_dataset(name, *, tfds_data_dir=None, tfr_file=None, seed=547):
  """Instantiates a data set and sets the random seed."""
  if name not in DATASETS:
    raise ValueError("Dataset %s is not available." % name)
  kwargs = {}

  if name == 'lsun' or name == 'tfork':
    # LsunDataset takes the path to the tf record, not a directory
    assert tfr_file is not None
    kwargs['tfr_file'] = tfr_file
  else:
    kwargs['tfds_data_dir'] = tfds_data_dir

  if name not in ['lsun', 'tfork', *SimpleDataset.DATASET_NAMES]:
    kwargs['seed'] = seed

  return DATASETS[name](**kwargs)
