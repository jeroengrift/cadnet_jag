import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from types import SimpleNamespace
from pipeline.split import SplitDataset

CFG = SimpleNamespace(
    DATASETS=SimpleNamespace(
        DATA_DIR='data/nl',
        IMAGES_DIR='images',
        SPLIT_FILE='data/nl/cad_reference/sample_tiles_10k_split.gpkg',
    )
)

if __name__ == "__main__":
    sd = SplitDataset(cfg=CFG)
    sd.split_dataset_file()
