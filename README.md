# Deep Learning Hip Dysplasia

A Deep Learning Classifier/Estimator for measuring hip properties and hip dysplasia

## Getting Starting 

You will need to have python installed with tensorflow v2.0 and a number of python libraries. The simplest method to do this is to install the [conda package manager](https://conda.io/miniconda.html) and run the following command:

```
conda create -n tf python=3.6 numpy=1.17.3 matplotlib scikit-image=0.15.0 tqdm
```

Then if you have a CUDA enabled GPU use: `pip install tensorflow-gpu`
otherwise use `pip install tensorflow`

Activate the virtual environment you created with `source activate tf` (Mac) or `activate tf` (PC)

# Usage

To run (while in the root directory) use the following command:

```
python main.py
```
