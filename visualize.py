from utils import show_tiff
from glob import glob 

#change the path according to requirement 
tif_path = "./data/training.tif"
output_path = "./data/visualize/images/"

show_tiff(tif_path, output_path)