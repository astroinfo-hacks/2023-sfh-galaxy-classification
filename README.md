# Galaxy classification from star formation history

## Rationale

A presentation was prepared for the pre-hack telecon. 
You can watch the [slides][slides] and the [presentation itself][telecon].

[slides]: https://docs.google.com/presentation/d/1smHllIigLpfG2uP7hLTpQNIABpR50r0Zqe2EbgzOJwk/edit?usp=sharing
[telecon]: https://u-paris.zoom.us/rec/share/ibQAB_HcRwoRFxrmne3RtWUnGp3xH_bqsS9oOG0vMHZEPJidfSASYsXzR_MzNCM.0GfrQ39bReZsAScg

## Dataset

The binned SFH can be found in the `data` folder.

## Notebook

- [Binned SFHs `visualitation`](Data-visluation.ipynb)
- [Time-series classification with `tslearn`](k-means_time-series-example.ipynb)

## Bibliography

- [Analyzing Astronomical Data with Machine Learning Techniques](https://arxiv.org/pdf/2302.11573.pdf)
  - Small review on the state of the art on astro-ML techniques.
- [K-Means Clustering: How It Works & Finding The Optimum Number Of Clusters In The Data](https://towardsdatascience.com/k-means-clustering-how-it-works-finding-the-optimum-number-of-clusters-in-the-data-13d18739255c)
  - Mathematical formulation, Finding the optimum number of clusters and a working example in Python 
- [Unsupervised classification of CIGALE galaxy spectra](https://arxiv.org/pdf/2205.09344.pdf)
  - SED fitting & Galaxy spectra calssification.
- [The complexity of galaxy populations](https://arxiv.org/pdf/1805.09904.pdf)
  - Galaxy classification on rest-frame magnitudes and spectroscopic redshifts.
- [Interpretable Time-series Classification on Few-shot Samples](https://arxiv.org/pdf/2006.02031.pdf)
  - Shapelets: Other approach on the time-series classification.
- [Evolution of the star formation with time (see chapter IV) ](https://people.lam.fr/buat.veronique/Veronique/Teaching_files/Lecture4.pdf)
  - Lecture from V.Buat@LAM
- [Quenched galaxy](http://astro.vaporia.com/start/quenchedgalaxy.html)
- [Fast (and slow) quenching channels](https://arxiv.org/pdf/1802.07628.pdf)
