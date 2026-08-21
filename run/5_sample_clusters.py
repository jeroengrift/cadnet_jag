import geopandas as gp
import pandas as pd
import os
import rasterio as rio
import numpy as np
import warnings
from types import SimpleNamespace
from shapely.geometry import box

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

CFG = SimpleNamespace(
    pixel_size_patch=25,
    sample_size_per_tile=30,
    input_dir='data/nl',
    overlap_50=True,
    land_file='data/nl/cad_reference/deellandschapPolygon.shp',
)


def run(cfg):
    land = gp.read_file(cfg.land_file).dissolve()
    collected_patches = gp.GeoDataFrame(columns=['cluster_type', 'cluster_id', 'image_id', 'urban', 'split', 'geometry'], crs='EPSG:28992')

    patch_file_location = (
        f'{cfg.input_dir}/reference/overlap_50/urban_{cfg.pixel_size_patch}.gpkg'
        if cfg.overlap_50
        else f'{cfg.input_dir}/reference/urban_{cfg.pixel_size_patch}.gpkg'
    )
    patch_file = gp.read_file(patch_file_location, layer=f'urban_{cfg.pixel_size_patch}', engine="pyogrio")

    unique_image_ids = (
        patch_file
        .assign(image_suffix=patch_file['image_id'].apply(lambda x: x.split('_')[3].split('.')[0]))
        [['image_suffix', 'split']]
        .drop_duplicates()
        .apply(tuple, axis=1)
        .tolist()
    )

    for idx, image_id in enumerate(unique_image_ids):
        img_id = image_id[0]
        split = image_id[1]
        tile_dir = f'{cfg.input_dir}/{split}/images/'

        print(f"[{idx}] Processing image_id: {img_id}")
        image_data = patch_file[patch_file['image_id'].str.contains(img_id)]
        image_tile = rio.open(os.path.join(tile_dir, f'{img_id}.tif'))

        image_tile_bounds = image_tile.bounds
        minx, miny, maxx, maxy = image_tile_bounds

        subframe_size = 500
        subframes = []
        x_coords = np.arange(minx, maxx, subframe_size)
        y_coords = np.arange(miny, maxy, subframe_size)

        for x in x_coords:
            for y in y_coords:
                if (x + subframe_size <= maxx) and (y + subframe_size <= maxy):
                    subframes.append(box(x, y, x + subframe_size, y + subframe_size))

        clusters_gdf = gp.GeoDataFrame({'geometry': subframes}, crs=image_tile.crs)
        clusters_gdf = clusters_gdf.sjoin(land, how='inner', predicate='intersects').reindex(columns=['geometry'])
        clusters_gdf = clusters_gdf.sample(n=cfg.sample_size_per_tile)

        patches_with_clusters = gp.sjoin(image_data, clusters_gdf, how='inner', predicate='within')
        patches_with_clusters = patches_with_clusters.rename(columns={'index_right': 'cluster_id'})

        urban_patch_counts = patches_with_clusters.groupby('cluster_id').agg(
            urban_patch_count=('urban', lambda x: (x == True).sum()),
            non_urban_patch_count=('urban', lambda x: (x == False).sum())
        ).reset_index()

        patches_urban = patches_with_clusters[patches_with_clusters['cluster_id'].isin(urban_patch_counts['cluster_id'])].copy()
        patches_urban = patches_urban.reindex(columns=['geometry', 'cluster_type', 'cluster_id', 'image_id', 'urban', 'split'])

        urban_count = patches_urban['urban'].sum()
        print(f"Urban patches count: {urban_count}")
        print(f"Non-urban patches count: {len(patches_urban) - urban_count}")

        collected_patches = pd.concat([collected_patches, patches_urban], ignore_index=True)

    urban_count = collected_patches['urban'].sum()
    print(f"Urban patches count: {urban_count}")
    print(f"Non-urban patches count: {len(collected_patches) - urban_count}")

    out_path = (
        f'{cfg.input_dir}/reference/overlap_50/urban_{cfg.pixel_size_patch}_clusters.gpkg'
        if cfg.overlap_50
        else f'{cfg.input_dir}/reference/urban_{cfg.pixel_size_patch}_clusters.gpkg'
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    collected_patches.to_file(out_path, layer=f'urban_{cfg.pixel_size_patch}_clusters', driver='GPKG')


if __name__ == "__main__":
    run(CFG)
