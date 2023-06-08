# Can Deep Learning help to forecast deforestation in the Amazonian Rainforest?
## Abstract
Deforestation is a major driver of climate change. Conservation projects aim to protect endangered regions. Researchers have proposed several ways to derive baseline counterfactuals for the project sites retrospectively. However, limited tools exist to forecast deforestation in advance, which would help to identify threatened sites. This study examines the effectiveness of Deep Learning (DL) models in predicting the location of deforestation 1-year ahead. The models yield unsatisfactory results for pixel-wise classification (F1: 0.274) at a 30 m resolution. When simplifying the task to binary classification of 1.5 km squared tiles, the models achieve moderate performance metrics (F1: 0.618). As main challenges, we see the noise in the data and underlying processes as well as the extreme class imbalance between deforested (minority) and non-deforested (majority) pixels. We conclude that, at the current state of research, DL models can only partially help in forecasting deforestation.

## Installation
### Locally (for data preparation and model training)
Create a new conda environment and install the following packages:
- python=3.9
- jupyter
- numpy
- matplotlib
- pandas
- geopandas
- rasterio
- pytorch torchvision -c pytorch-nightly
- pytorch-lightning -c conda-forge
- tensorboard

### Euler Cluster (only for model training)
Create a new conda environment and install the following packages:
- python=3.8.5
- pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
- pytorch-lightning
- tensorboard
- rasterio

## Execution
To train a model, download the processed dataset from [here](https://polybox.ethz.ch/index.php/s/izGOJlS52qj9zWj) and place it in the data folder. 
Adjust the ````config.json```` file to your wishes. Check out the config files used for our model training, to see the different options. Ensure that in ````train_model.py```` the config file is loaded correctly. Now, the model should start training. To train a model on Euler, you can use the ````submit.sh```` script, and specify the config file as an argument.

To make predictions with a trained model, use your own or download our models from [here](https://polybox.ethz.ch/index.php/s/izGOJlS52qj9zWj) and place it in the models folder. Then, set the model name and properties in the file ````predict_model.py````. The predictions will be saved in the same folder as the model.

## Code Structure
Most files are provided through this repository. Additionally, we provide the processed data, the described models and our reports [here](https://polybox.ethz.ch/index.php/s/izGOJlS52qj9zWj). If you want to access the raw data, refer to: [Mapbiomas](https://mapbiomas.org/download) and [FABDEM](https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn). If you need any other files, feel free to contact us.

```
.
├── configs                     # Contains example config file to run pipeline
├── data                        
│   ├── processed               # Holds processed data in pytorch format
│   └── raw                     # Holds untouched raw data, downloaded as described
│       ├── biomas              # Store all Mapbiomas data like this: 
│       │                         /amazonia/30m/deforestation/filename.tif
│       └── fabdem              # Store all FABDEM mosaic folders in here 
├── models                      # Contains the trained models described in the report
├── notebooks                   # Jupyter notebooks for data analysis
│   ├── data_layers_...ipynb    # Notebook to visualize data layers and distance to deforestation
│   ├── deforestation_...ipynb  # Notebook calculating deforestation rates for different data layers
│   ├── feature_normal...ipynb  # Notebook investigating effect of feature normalization
│   ├── input_output_...ipynb   # Notebook visulaizing input and output at different scales (earlier version)
│   ├── naive_baseline.ipynb    # Notebook implementing naive baseline
│   ├── predictions.ipynb       # Notebook to visualize predictions
│   ├── random_forest.ipynb     # Notebook implementing random forest baseline
│   └── satellite.ipynb         # Notebook to prepare satellite data for visualization purposes
├── reports                     # Contains deforestation rates and plots
├── src                         # Main code
│   ├── features                # Code to turn raw data into processed dataset
│       └── build_features.py   # Run this file to create processed dataset
│   └── models                  # Code for model training and prediction
│       ├── utils               # Helpers: cnn, dataset, model
│       ├── predict_model.py    # Run this file to make predictions with a trained model
│       └── train_model.py      # Run this file to train a model
└──  submit.sh                  # Shell script to submit training jobs to Euler
```



