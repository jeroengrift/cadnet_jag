import os
import re
import shutil
import geopandas as gp
import glob
from pathlib import Path
import numpy as np
from itertools import cycle


class SplitDataset():
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_dir = os.path.join(cfg.DATASETS.DATA_DIR, 'train', cfg.DATASETS.IMAGES_DIR)
        self.validate_dir = os.path.join(cfg.DATASETS.DATA_DIR, 'validate', cfg.DATASETS.IMAGES_DIR)
        self.test_dir = os.path.join(cfg.DATASETS.DATA_DIR, 'test', cfg.DATASETS.IMAGES_DIR)
        dirs = [self.train_dir, self.validate_dir, self.test_dir]

        for folder in dirs:
            Path(folder).mkdir(parents=True, exist_ok=True)
        
    def split_dataset_file(self):
        tiles_gdf = gp.read_file(self.cfg.DATASETS.SPLIT_FILE)
        tiles_gdf = tiles_gdf[['filename_10k', 'split']]
        tiles_gdf['filename_10k'] = tiles_gdf['filename_10k']\
            .str.replace('tile-', '')\
            .str.replace('.tif', '')\
            .str.replace('_planet', '')\
            .str.replace('_sentinel', '')\
            .str.replace('_superview', '')\
            .apply(lambda x: re.sub(r'(\d+)\.\d+', lambda m: str(round(float(m.group(0)))), x))

        for index, row in tiles_gdf.iterrows():
            key = row['filename_10k']
            img_src_str = os.path.join(self.cfg.DATASETS.DATA_DIR, self.cfg.DATASETS.IMAGES_DIR, f'tile-{key}.tif')
            if not os.path.exists(img_src_str):
                img_src_str = os.path.join(self.cfg.DATASETS.DATA_DIR, self.cfg.DATASETS.IMAGES_DIR, f'{key}.tif')
            if not os.path.exists(img_src_str):
                print(f"Skipping missing image: {key}.tif")
                continue

            if row['split'] == 'train':
                shutil.copyfile(img_src_str, os.path.join(self.train_dir, f'{key}.tif'))

            if row['split'] == 'validate' or row['split'] == 'val':
                shutil.copyfile(img_src_str, os.path.join(self.validate_dir, f'{key}.tif'))

            if row['split'] == 'test':
                shutil.copyfile(img_src_str, os.path.join(self.test_dir, f'{key}.tif'))
                    