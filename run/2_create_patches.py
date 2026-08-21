import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import multiprocessing as mp
from types import SimpleNamespace
from pipeline.patches import CreatePatches

CFG = SimpleNamespace(
    DATA_DIR='data/nl',
    IMAGES_DIR="images",
    PATCHES_DIR="patches_25",
    IMG_SIZE=512,
    TEST_OVERLAP_50=True,
    DOWNSAMPLE_FACTOR=.32,
)

if __name__ == "__main__":
    cp = CreatePatches(CFG)
    cp.create_patch()

