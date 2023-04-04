from utils import show_tiff
from glob import glob 
import os 

#change the path according to requirement 
tif_path = "./data/testing.tif"
output_path = "./data/visualize/images/"
os.makedirs(output_path, exist_ok =True)
show_tiff(tif_path, output_path)
