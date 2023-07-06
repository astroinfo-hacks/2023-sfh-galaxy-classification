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

## Pipeline

[![](https://mermaid.ink/img/pako:eNqNk01r20AQhv_KMIeQCO_Bdj7Ah5aQkhoa-RCHFNotZCKN6yWrXbG7whEh_z0rS5Ytk0J10r6aZz7e0b5hZnPGGa603WRrckEagGXgcvxb4lOS3NMGlrdznyTNh92zoEBa1fQk8Y80O2TSIg-OjF9ZV3wCpvQ6ZKZdmcrAD5FyJIfAtQn-qMx5i3znADe68oEd3NEz63-TLXex4ww7Cgx5HAI8hyNuTkYRnMBNdEPXw9KXbYq00kGJTJP3MJgueqljU1FXK8UuSeB_M1-1ma_LUtdgDTgmve0wSfaBhc3Hp6cS728lnp11yqRRlo_pgTRtpMWiVfY1xkKAxLndQEGmhmBf2PivEkGIL9v99YscBmaty4eh035_nXDeb6cTLnrbO-GyN1EI0UxydJ4cnacH59jNzzUFUB7CmuGZfWjmZH3Q0lUzKI6wYFeQyuM__dZkkBiJgiXO4mtO7kWiNO8xjqpgl7XJcBZcxSOsymg3f1P011GBsxVpH1XOVbAubS_J9q6MsCTzy9pdzPsHdlb_aA?type=png)](https://mermaid.live/edit#pako:eNqNk01r20AQhv_KMIeQCO_Bdj7Ah5aQkhoa-RCHFNotZCKN6yWrXbG7whEh_z0rS5Ytk0J10r6aZz7e0b5hZnPGGa603WRrckEagGXgcvxb4lOS3NMGlrdznyTNh92zoEBa1fQk8Y80O2TSIg-OjF9ZV3wCpvQ6ZKZdmcrAD5FyJIfAtQn-qMx5i3znADe68oEd3NEz63-TLXex4ww7Cgx5HAI8hyNuTkYRnMBNdEPXw9KXbYq00kGJTJP3MJgueqljU1FXK8UuSeB_M1-1ma_LUtdgDTgmve0wSfaBhc3Hp6cS728lnp11yqRRlo_pgTRtpMWiVfY1xkKAxLndQEGmhmBf2PivEkGIL9v99YscBmaty4eh035_nXDeb6cTLnrbO-GyN1EI0UxydJ4cnacH59jNzzUFUB7CmuGZfWjmZH3Q0lUzKI6wYFeQyuM__dZkkBiJgiXO4mtO7kWiNO8xjqpgl7XJcBZcxSOsymg3f1P011GBsxVpH1XOVbAubS_J9q6MsCTzy9pdzPsHdlb_aA)

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
