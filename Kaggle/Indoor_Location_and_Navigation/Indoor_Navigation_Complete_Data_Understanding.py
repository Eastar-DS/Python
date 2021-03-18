# Indoor Navigation: Complete Data Understanding from code category in Kaggle

#pip install opencv-python
#pip install plotly
#pip install wandb
#pip install kaggle

#1
# CPU libraries
import os
import json
import glob
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objs as go

from PIL import Image, ImageOps
from skimage import io
from skimage.color import rgba2rgb, rgb2xyz
from tqdm import tqdm
from dataclasses import dataclass
from math import floor, ceil

mycolors = ["#797D62", "#9B9B7A", "#D9AE94", "#FFCB69", "#D08C60", "#997B66"]




#2
def show_values_on_bars(axs, h_v="v", space=0.4):
    '''Plots the value at the end of the a seaborn barplot.
    axs: the ax of the plot
    h_v: weather or not the barplot is vertical/ horizontal'''
    
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y, format(value, ','), ha="center") 
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, format(value, ','), ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


#3
import wandb
os.environ["WANDB_SILENT"] = "true"
'''
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
personal_key_for_api = user_secrets.get_secret("wandb-key")
'''
#! wandb login ddc25912c8fced10b40e54aff2c1d5f5dfafcf96

# Initialize new project
### project - the name of the overajj project (like the name of the repo in GitHub)
### name/experiment - the name of the run (we'll have multiple runs in 1 project)
#run = wandb.init(project="indoor-location-kaggle", name="data-understanding")




#4
# Get path to all TRAIN & TEST files
train_paths = glob.glob('train/*/*/*')
test_paths = glob.glob('test/*')
sites = glob.glob('metadata/*')

print("No. Files in Train: {:,}".format(len(train_paths)), "\n" +
      "No. Files in Test: {:,}".format(len(test_paths)), "\n" +
      "Total Sites (metadata): {:,}".format(len(sites)))

#  Log to W&B in "data-understanding" experiment
wandb.log({'No. Files in Train': len(train_paths), 
           'No. Files in Test:' : len(test_paths),
           'Total Sites (metadata)' : len(sites)})




#5 Reading in the data
# How 1 path looks
# base = '../input/indoor-location-navigation'
# path = f'{base}/train/5cd56b5ae2acfd2d33b58549/5F/5d06134c4a19c000086c4324.txt'
path = f'train/5cd56b5ae2acfd2d33b58549/5F/5d06134c4a19c000086c4324.txt'

#open에 encoding을 다시해주어 오류해결
with open(path, 'r', encoding='UTF8') as p:
    lines = p.readlines()

print("No. Lines in 1 example: {:,}". format(len(lines)), "\n" +
      "Example (5 lines): ", lines[0:5])




#6 How to use a GitHub repo on Kaggle
'''
!cp -r indoor-location-competition-20-master/indoor-location-competition-20-master/* ./

# Import custom function from the repository
from io_f import read_data_file

# Read in 1 random example
sample_file = read_data_file(path)

# You can access the information for each variable:
print("~~~ Example ~~~")
print("acce: {}".format(sample_file.acce.shape), "\n" +
      "acacce_uncalice: {}".format(sample_file.acce_uncali.shape), "\n" +
      "ahrs: {}".format(sample_file.ahrs.shape), "\n" +
      "gyro: {}".format(sample_file.gyro.shape), "\n" +
      "gyro_uncali: {}".format(sample_file.gyro_uncali.shape), "\n" +
      "ibeacon: {}".format(sample_file.ibeacon.shape), "\n" +
      "magn: {}".format(sample_file.magn.shape), "\n" +
      "magn_uncali: {}".format(sample_file.magn_uncali.shape), "\n" +
      "waypoint: {}".format(sample_file.waypoint.shape), "\n" +
      "wifi: {}".format(sample_file.wifi.shape))

~~~ Example ~~~
acce: (20515, 4) 
acacce_uncalice: (0,) 
ahrs: (20906, 4) 
gyro: (20906, 4) 
gyro_uncali: (20906, 4) 
ibeacon: (392, 3) 
magn: (20906, 4) 
magn_uncali: (20906, 4) 
waypoint: (31, 3) 
wifi: (764, 5)
'''



#7 Sites
def show_site_png(site):
    '''This functions outputs the visualization of the .png images available
    in the metadata.
    sites: the code coresponding to 1 site (or building)'''
    
    #base = '../input/indoor-location-navigation'
    site_path = f"metadata/{site}/*/floor_image.png"
    floor_paths = glob.glob(site_path)
    n = len(floor_paths)

    # Create the custom number of rows & columns
    ncols = [ceil(n / 3) if n > 4 else 4][0]
    nrows = [ceil(n / ncols) if n > 4 else 1][0]

    plt.figure(figsize=(16, 10))
    plt.suptitle(f"Site no. '{site}'", fontsize=18)

    # Plot image for each floor
    for k, floor1 in enumerate(floor_paths):
        plt.subplot(nrows, ncols, k+1)

        image = Image.open(floor1)
        image = ImageOps.expand(image, border=15, fill=mycolors[5])

        plt.imshow(image)
        plt.axis("off")
        # title = floor1.split("/")[5]
        # plt.title(title, fontsize=15)
        
        # Log to W&B in "data-understanding" experiment
        wandb.log({"Site Floors Example": plt})

# Let's observe 1 example
# site = '5cd56b64e2acfd2d33b59246'
show_site_png(site='5cd56b64e2acfd2d33b592b3')


#8
































