import os

import tensorflow_datasets as tfds

from sfh_classif.dataset import HorizonAGN

# Ensure tfds does not try to fetch the dataset from the internet
os.environ['NO_GCE_CHECK'] = 'true'
tfds.core.utils.gcs_utils._is_gcs_disabled = True

_JZ_TF_DATASET_DIR = "/gpfsscratch/rech/owt/commun/galaxy_classification/tf_datasets"

# The following line will either load the TensorFlow dataset if existing
# or it will create the dataset from the _generate_examples method
dset = tfds.load("HorizonAGN", split=tfds.Split.TRAIN, data_dir=_JZ_TF_DATASET_DIR)