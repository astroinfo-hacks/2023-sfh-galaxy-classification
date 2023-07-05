import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from sfh_classif.preprocessing import load_sfh_data
from sfh_classif.preprocessing import create_regular_mass_grid
from sfh_classif.preprocessing import resample_in_time


_JZ_ROOT_PATH = "/gpfsscratch/rech/owt/commun/galaxy_classification/SFH_all/downloads/manual"
_JZ_FILE_LIST = "/gpfsscratch/rech/owt/commun/galaxy_classification/sfh_filenames.txt"

_DESCRIPTION = "SFH from Horizon AGN processed by Rafael Arango"
_URL = "https://github.com/astroinfo-hacks/2023-sfh-galaxy-classification"
_CITATION = ""

# Maximum number of 1 My timesteps in the entire dataset (700 000 files)
N_TIMESTEPS = 12200
# Binning of My to reduce the size of the data
N_YEARS = 10
# Size of the output vectors
OUTPUT_SIZE = N_TIMESTEPS // N_YEARS


class HorizonAGN(tfds.core.GeneratorBasedBuilder):
  """Horizon AGN SFH galaxy dataset"""  

  VERSION = tfds.core.Version("1.0.0")
  RELEASE_NOTES = {'1.0.0': 'Initial release.',}
  MANUAL_DOWNLOAD_INSTRUCTIONS = "Nothing to download. Dataset was generated at first call."
  
  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        homepage=_URL,
        citation=_CITATION,
        # Two features: image with 3 channels (stellar light, velocity map, velocity dispersion map)
        #  and redshift value of last major merger
        features=tfds.features.FeaturesDict({
            "time": tfds.features.Tensor(shape=(OUTPUT_SIZE,), dtype=tf.float32),
            "mass": tfds.features.Tensor(shape=(OUTPUT_SIZE,), dtype=tf.float32),
            'cumulative_mass': tfds.features.Tensor(shape=(OUTPUT_SIZE,), dtype=tf.float32),
            "max_time": tf.int32,
            'filename': tf.string
        }),
    )

  def _split_generators(self, dl):
    """Returns generators according to split"""
    return {tfds.Split.TRAIN: self._generate_examples()}

  def _generate_examples(self):
    """Yields examples."""
    with open(_JZ_FILE_LIST) as f:
      filenames = f.read().splitlines()

    for i, filename in enumerate(filenames):
        filepath = os.path.join(_JZ_ROOT_PATH, filename)
        time, mass = load_sfh_data(filepath)
        grid_mass = create_regular_mass_grid(time, mass, N_TIMESTEPS)
        binned_time, binned_mass = resample_in_time(grid_mass, N_YEARS)
        res = {
           "time": binned_time.astype(np.float32),
           "mass": binned_mass.astype(np.float32),
           "cumulative_mass": np.cumsum(binned_mass).astype(np.float32),
           "max_time": (np.floor(time[-1] / 10)).astype(np.int32),
           "filename": filename,
        }
        yield i, res
