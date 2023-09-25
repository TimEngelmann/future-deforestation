# Can Deep Learning help to forecast deforestation in the Amazonian Rainforest?
## Abstract
Deforestation is a major driver of climate change. To mitigate deforestation, carbon offset projects aim to protect forest areas at risk. However, existing literature shows that most projects have substantially overestimated the risk of deforestation, thereby issuing carbon credits without equivalent emissions reductions. In this study, we examine if the spread of deforestation can be predicted ex-ante using Deep Learning (DL) models. Our input data includes past deforestation development, slope information, land use, and other terrain- and soil-specific covariates. Testing predictions 1-year ahead, we find that our models only achieve low levels of predictability. For pixel-wise classification at a 30 m resolution, our models achieve an F1 score of 0.263. Only when substantially simplifying the task to predicting if any level of deforestation occurs within a 1.5 km squared tile, the model results improve to a moderate performance (F1: 0.608). We conclude that, based on our input data, slope information, and other common covariates, deforestation cannot be predicted accurately enough to justify the ex-ante issuance of carbon credits for forest conservation projects. As main challenges, there is the extreme class imbalance between pixels that are deforested (minority) and not deforested (majority) as well as the omittance of social, political, and economic drivers of deforestation.

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

### Cluster (only for model training)
Create a new conda environment and install the following packages:
- python=3.8.5
- pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
- pytorch-lightning
- tensorboard
- rasterio

## Execution
To train a model, request the processed dataset and place it in the data folder. 
Adjust the ````config.json```` file to your wishes. Check out the config files used for our model training, to see the different options. Ensure that in ````train_model.py```` the config file is loaded correctly. Now, the model should start training. To train a model on the Cluster, you can use the ````submit.sh```` script, and specify the config file as an argument.

To make predictions with a trained model, use your own or request our models and place it in the models folder. Then, set the model name and properties in the file ````predict_model.py````. The predictions will be saved in the same folder as the model.

## Code Structure
Most files are provided through this repository. Additionally, we are happy to share the processed data, the described models and our reports upon request. If you want to access the raw data, refer to: [Mapbiomas](https://mapbiomas.org/download) and [FABDEM](https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn). If you have any other questions, feel free to contact us.

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
└──  submit.sh                  # Shell script to submit training jobs to Cluster
```

## Experiments Nomen Clature
As this repository allows to run a wider variety of experiments, than described in the workshop paper, we want to clarify the naming of the experiments. The following table shows the naming of the experiments in the workshop paper and the corresponding ``config.json`` file in this repository.
````
Pixel-wise classification | 2D-CNN  ->  config_pixel_classification.json
Pixel-wise classification | UNet    ->  config_tile_segmentation.json
Tile-wise classification  | 2D-CNN  ->  config_tile_classification.json
````


