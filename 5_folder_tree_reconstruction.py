# This is the last step of the media processing, even after the QA work
# The code below reconstructs the original folder structure of all the files.

import pandas as pd
import shutil

csv_location = 'C:\\D\\DSU\\github\\gather\\mapping.csv' # replace it with the csv file created in the gather_images.py script

df = pd.read_csv(csv_location)

for i, row in df.iterrows():
    shutil.move(row['new_path'], row['old_path'])
