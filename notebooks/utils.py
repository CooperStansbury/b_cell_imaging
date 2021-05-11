import os
import pandas as pd
import numpy as np
from skimage import io


class FileLoader():
    """ Utility class to load images into a useable structure """

    def parse_filename(self, filename):
        return {
            'filename' : filename,
            'group' : filename.split(" ")[0],
            'day' : int(filename.split("Day ")[1].split("_")[0]),
            'channel' : filename.split("_")[1].split(".")[0],
            'channel_color' : self.channel_colors[filename.split("_")[1].split(".")[0]]
        }

    def load_imgs_from_dir(self):
        new_rows = []
        for f in os.listdir(self.path):
            file_metadata = self.parse_filename(f)

            full_path = f"{self.path}{f}"
            file_metadata['image'] = io.imread(full_path)
            new_rows.append(file_metadata)

        return pd.DataFrame(new_rows)

    def __init__(self, dir_path):

        self.channel_colors = {
            'Ch2-T1' : 'red', 
            'ChS2-T2' : 'yellow', 
            'Ch1-T4' : 'cyan', 
            'ChS1-T3' : 'blue' 
        }
        self.path = dir_path
        self.df = self.load_imgs_from_dir()
