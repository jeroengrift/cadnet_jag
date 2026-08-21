

# From pixels to vectorized cadastral boundaries: Deep learning-based automated delineation of property boundaries in the Netherlands

This repository contains the code for CadNET, a deep learning model for automated delineation of cadastral boundaries from aerial imagery. More information can be found in the paper: https://www.sciencedirect.com/science/article/pii/S1569843226004309

To run the CadNet model on the CadastreVision dataset, follow the steps below:

## Set up conda environment

```bash
conda env create -f environment.yml
conda activate cad
```

---

## Data setup

### Images

This experiment uses 8 cm/px aerial imagery resampled to 25 cm/px. The 8 cm data is not included in the CadastreVision benchmark due to data size. Two options:

- **Use the 25 cm tiles** already in the [CadastreVision benchmark dataset](https://github.com/jeroengrift/cadastrevision) directly.
- **Use the original 8 cm imagery** by downloading it manually from https://basisdata.nl/hwh-portal/download/index.html and resampling to 25 cm.

Put all aerial tiles flat in:

```text
data/nl/images/
```

### Reference data

The preprocessing scripts (steps 1-5 below) require reference data. Ensure the following files are in `data/cad_reference/`:

- `sample_tiles_10k_split.gpkg`: tile split configuration (train/validate/test)
- `brk_reference.gpkg`: cadastral boundaries (BRK)
- `brt_bebouwde_kom.gpkg`: urban/rural classification boundaries
- `deellandschapPolygon.shp`: landscape parcels for cluster sampling

the first three dataset can be downlaoded from https://dansdataportal.nl/dataset.xhtml?persistentId=doi:10.17026/PT/OS3OWX

The fourth is available here: https://nationaalgeoregister.nl/geonetwork/srv/dut/catalog.search#/metadata/9a9cef3a-2dfc-4aa8-b248-f73f4064d7ad

### Split dataset

Distributes tiles into `train/`, `validate/`, and `test/` folders based on `cad_reference/sample_tiles_10k_split.gpkg`. Edit `CFG` at the top of the script, then run:

```bash
python run/1_split_dataset.py
```

Produces `train/images/`, `validate/images/`, `test/images/`.

### Create patches

Slice the tiles into 512×512 patches. Edit `CFG` at the top of the script, then run:

```bash
python run/2_create_patches.py
```

Outputs go to `train/patches_25/images/`, `validate/patches_25/images/`, `test/patches_25/images/`.

### Generate reference files

Clips BRK cadastral reference lines to each patch. Edit `CFG` at the top of the script (`pixel_size_patch`, `overlap`), then run:

```bash
python run/3_patch_reference.py
```

Produces:

```text
reference/reference_nl_ps_25_brk.gpkg          # overlap: False
reference/overlap_50/reference_nl_ps_25_brk.gpkg  # overlap: True
```

### Generate urban/rural labels

Labels each patch as urban or rural. Requires patches and `cad_reference/brt_bebouwde_kom.gpkg`. Edit `CFG` at the top of the script, then run:

```bash
python run/4_label_urban.py
```

Produces `reference/urban_25.gpkg` (or `reference/overlap_50/urban_25.gpkg`).

### Generate cluster files

Samples spatial clusters per tile for training/evaluation grouping. Requires `urban_25.gpkg` and `cad_reference/deellandschapPolygon.shp`. Edit `CFG` at the top of the script, then run:

```bash
python run/5_sample_clusters.py
```

Produces `reference/urban_25_clusters.gpkg` (or `reference/overlap_50/urban_25_clusters.gpkg`).


---

## Running

### Training

```bash
python run/6_train.py --config config/cadnetv2/cadnet_brk_swinunet_multi_coa_v2_train.yaml
```

### Batch prediction

```bash
python run/7_predict_batch.py \
    --input_dir <config_dir> \
    --cluster_file data/nl/reference/overlap_50/urban_25_clusters.gpkg \
    --device cuda:0 \
    --num_workers 4 \
    --write_mosaic_raster true \
    --write_raster_stats true \
    --write_mosaic_vector true \
    --write_vector_stats true
```

`--input_dir` should contain a YAML config with `MODEL.RUN_TYPE: predict` and `MODEL.TRAINED_MODEL` pointing to the checkpoint filename. The config controls the backbone, `USE_MULTI` / `USE_COA` / `USE_BRK` flags and which outputs to write.

### Config

```text
config/cadnetv2/cadnet_brk_swinunet_multi_coa_v2_train.yaml   # training
config/cadnetv2/cadnet_brk_swinunet_multi_coa_v2.yaml         # prediction
```

---

## Output

Results are written to:

```text
./output/jag_2026/model-CadNET_bb-<backbone>/
```

Each run saves checkpoints, loss logs, mosaic raster predictions, raster metrics, vector predictions, and vector metrics.

---

## Citation
For citing this paper in publications, use:

```bibtex
@article{GRIFT2026105514,
title = {From pixels to vectorized cadastral boundaries: Deep learning-based automated delineation of property boundaries in the Netherlands},
journal = {International Journal of Applied Earth Observation and Geoinformation},
volume = {153},
pages = {105514},
year = {2026},
issn = {1569-8432},
doi = {https://doi.org/10.1016/j.jag.2026.105514},
url = {https://www.sciencedirect.com/science/article/pii/S1569843226004309},
author = {Jeroen Grift and Claudio Persello and Mila Koeva},
keywords = {Cadastral boundary extraction, Aerial imagery, Deep learning, Land administration, Remote sensing, Vectorization}
}
```

