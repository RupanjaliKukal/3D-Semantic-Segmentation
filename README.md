# 3D Semantic Segmentation

## Environment

## Downloading Data
Download the Electron Microscopy Dataset as multipage .tif files from [here](https://www.epfl.ch/labs/cvlab/data/data-em/)

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
python test.py
```

