# ROCKAD: Transferring ROCKET to Whole Time Series Anomaly Detection

ROCKAD is a kernel-based approach proposed for semi-supervised whole time series anomaly detection. ROCKAD employs ROCKET as an unsupervised feature extractor, training a single and ensemble of k-nearest neighbors anomaly detectors to deduce an anomaly score. ROCKAD has been systematically evaluated for univariate time series and has also been demonstrated to be applicable to multivariate time series.


For a comprehensive understanding of how to use ROCKAD, please refer to our detailed guide, [How to use ROCKAD](./docs/How_to_use_ROCKAD.ipynb).

Further information and complete results can be found on our dedicated website: [ml-and-vis.org/rockad](https://ml-and-vis.org/rockad).

---

## Getting Started: Installation Instructions 

ROCKAD is compatible with both Windows and Linux platforms. If you encounter any issues during the installation process, please feel free to get in touch with us. 

### Windows and Linux 

Installing ROCKAD on Windows and Linux should generally be trouble-free. If you do face any issues, please don't hesitate to contact us. 

### Mac/Apple Silicon

For users with Apple products featuring the Apple Silicon (M1 Chips and later versions), please note that some dependencies from the "all_extras" package may not be compatible, particularly numba. We recommend installing the dependencies using the "macm1-requirements.txt" file.

Additional steps may include setting up your Python environment using Python version 3.9.

For more detailed instructions, please refer to the [latest sktime documentation](https://www.sktime.net/en/latest/installation.html).
