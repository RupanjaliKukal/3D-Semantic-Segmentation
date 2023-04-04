# 3D Semantic Segmentation

## Environment
You can find the dependencies in `environment.yml` file. They can be downloaded by running 

```
conda env create -f environment.yml
```

## Downloading Data
Download the Electron Microscopy Dataset as multipage .tif files from [here](https://www.epfl.ch/labs/cvlab/data/data-em/) in the `./data` directory. 

## Visualizing Data 
Visualize the .tif files by running 

```
python visualize.py
```

The images will be saved in `./data` directory.


## Training 
Run the training by running 

```
python main.py --NUM_EPOCHS=50
```

## Testing 

Run the training by running 

```
python test.py --RUN_ID=12
```

Pre-trained model is available in `./checkpoints` directory.
