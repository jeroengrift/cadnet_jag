import os
import warnings
from types import SimpleNamespace

import geopandas as gp
import numpy as np
import pandas as pd
import rasterio as rio
from shapely.geometry import box
from tqdm import tqdm

warnings.filterwarnings("ignore")

CFG = SimpleNamespace(
    crs=28992,
    pixel_size_patch=25,
    input_dir='nl',
    reference_dir='data/nl/cad_reference',
    data_root='data',
    overlap={'train': False, 'validate': False, 'test': True},
    urban_file='data/nl/cad_reference/brt_bebouwde_kom.gpkg',
    use_visibility=True,
)


def run(cfg):
    in_dir = os.path.join(cfg.data_root, cfg.input_dir)
    base_patches_dir = f'patches_{cfg.pixel_size_patch}'

    if isinstance(cfg.overlap, dict):
        out_dir = os.path.join(in_dir, 'reference', 'overlap_50') if any(cfg.overlap.values()) else os.path.join(in_dir, 'reference')
    else:
        out_dir = os.path.join(in_dir, 'reference', 'overlap_50') if cfg.overlap else os.path.join(in_dir, 'reference')
    os.makedirs(out_dir, exist_ok=True)

    reference_file = os.path.join(cfg.reference_dir, 'brk_reference.gpkg')

    def _patches_dir(split):
        flag = cfg.overlap[split] if isinstance(cfg.overlap, dict) else cfg.overlap
        if flag:
            overlap_dir = os.path.join(in_dir, split, f'{base_patches_dir}_overlap_50', 'images')
            if os.path.isdir(overlap_dir):
                return f'{base_patches_dir}_overlap_50'
        return base_patches_dir

    img_root = {
        split: os.path.join(in_dir, split, _patches_dir(split), 'images')
        for split in ('train', 'validate', 'test')
    }

    urban = gp.read_file(cfg.urban_file, engine="pyogrio") if cfg.urban_file is not None else None

    reference_gdf = gp.read_file(reference_file, engine="pyogrio")

    reference_gdf = reference_gdf.set_crs(cfg.crs, allow_override=True)
    sindex = reference_gdf.sindex
    out_gdfs = []

    for split, split_root in img_root.items():
        images = sorted(os.listdir(split_root))
        print(split, len(images))

        for image in tqdm(images):
            img_path = os.path.join(split_root, image)
            try:
                with rio.open(img_path) as ds:
                    bounds = ds.bounds
            except Exception:
                continue

            clip_mask = box(*bounds)

            try:
                candidate_idx = list(sindex.query(clip_mask, predicate="intersects"))
            except Exception:
                candidate_idx = list(sindex.intersection(clip_mask.bounds))

            if len(candidate_idx) == 0:
                continue

            subset = reference_gdf.iloc[candidate_idx].copy()
            if subset.empty:
                continue

            subset['image_id'] = image
            subset = subset.clip(clip_mask)

            if urban is not None:
                subset = subset.sjoin(urban, how='left', predicate='intersects')
                subset['urban'] = ~subset.BBK_ID.isnull()
            else:
                subset['urban'] = False

            if not cfg.use_visibility:
                subset['visible'] = True
            else:
                subset['visible'] = np.where(subset.visible == 'True', True, False)

            subset = subset[subset.geometry.type == 'LineString']
            if subset.empty:
                continue

            subset = subset.reindex(columns=[
                col for col in ['filename_10k', 'image_id', 'visible', 'urban', 'geometry'] if col in subset.columns
            ])
            subset['split'] = split
            out_gdfs.append(subset)

    if len(out_gdfs) == 0:
        print("No references found.")
    else:
        references_df = gp.GeoDataFrame(pd.concat(out_gdfs, ignore_index=True)).set_crs(cfg.crs)
        references_df.to_file(
            os.path.join(out_dir, f'reference_{cfg.input_dir}_ps_{cfg.pixel_size_patch}_brk.gpkg'),
            driver='GPKG',
        )


if __name__ == "__main__":
    run(CFG)
