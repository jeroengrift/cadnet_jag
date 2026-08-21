import os
import warnings
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import rasterio as rio
from shapely.geometry import box
from tqdm import tqdm

warnings.filterwarnings("ignore")

CFG = SimpleNamespace(
    pixel_size_patch='25',
    input_dir='data/nl',
    urban_gpkg='data/nl/cad_reference/brt_bebouwde_kom.gpkg',
    urban_layer='brk_bebouwde_kom',
    overlap_50=True,
)


def run(cfg) -> None:
    patches_dir = f'patches_{cfg.pixel_size_patch}'
    train_img_root = os.path.join(cfg.input_dir, 'train', patches_dir, 'images')
    val_img_root = os.path.join(cfg.input_dir, 'validate', patches_dir, 'images')
    if cfg.overlap_50:
        overlap_test_dir = os.path.join(cfg.input_dir, 'test', f'patches_{cfg.pixel_size_patch}_overlap_50', 'images')
        patches_dir = f'patches_{cfg.pixel_size_patch}_overlap_50' if os.path.isdir(overlap_test_dir) else patches_dir
    test_img_root = os.path.join(cfg.input_dir, 'test', patches_dir, 'images')
    img_root = {'train': train_img_root, 'validate': val_img_root, 'test': test_img_root}

    out_dir = os.path.join(cfg.input_dir, 'reference') if not cfg.overlap_50 else os.path.join(cfg.input_dir, 'reference', 'overlap_50')
    os.makedirs(out_dir, exist_ok=True)

    urban = gpd.read_file(cfg.urban_gpkg, layer=cfg.urban_layer, engine="pyogrio")
    urban = urban.to_crs("EPSG:28992")
    sindex = urban.sindex

    records = []
    for split, split_root in img_root.items():
        split_path = Path(split_root)
        if not split_path.exists():
            print(f"Skipping missing split directory: {split_path}")
            continue

        images = sorted([p for p in split_path.iterdir() if p.is_file()])
        print(split, len(images))

        for p in tqdm(images, desc=split):
            with rio.open(str(p)) as src:
                bounds = src.bounds

            clip_mask = box(*bounds)
            candidate_idx = list(sindex.intersection(bounds))
            is_urban = urban.iloc[candidate_idx].intersects(clip_mask).any() if candidate_idx else False

            records.append({
                "image_id": p.name,
                "urban": bool(is_urban),
                "split": split,
                "geometry": clip_mask,
            })

    if records:
        gdf_patch = gpd.GeoDataFrame(records, crs="EPSG:28992")
        out_file = os.path.join(out_dir, f"urban_{cfg.pixel_size_patch}.gpkg")
        gdf_patch.to_file(out_file, driver="GPKG")


if __name__ == "__main__":
    run(CFG)
