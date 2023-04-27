# Forecasting deforestation

# Packages
conda install python=3.9
conda install jupyter
conda install numpy
conda install matplotlib
conda install pandas
conda install geopandas
conda install rasterio
conda install pytorch torchvision -c pytorch-nightly
conda install pytorch-lightning -c conda-forge
conda install tensorflow

# Euler Cluster
python -m venv sp_venv
source sp_venv/bin/activate
pip install numpy
pip install torch torchvision
pip install pytorch-lightning
pip install tensorflow

pip install "jax[cuda11_cudnn805]==0.3.10" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.htmlpip 


pip install typing-extensions>=4.0.0

# Euler Cluster new
conda create -n sp
conda install python=3.8.5
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pytorch-lightning
conda install tensorboard
conda install rasterio

# Dataset
filter_px = 50

percentage of loss dropped:  0.0400
Deforestation percentage:  0.0075
nr train points:  180212
nr val points:  45053

