import numpy as np
import os
import geopandas as gp
import torch.utils.data
import rasterio as rio
import pandas as pd
import warnings
import sys

from shapely.geometry import box
from rasterio.features import rasterize
from skimage import io
from torch.utils.data.dataloader import default_collate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
dtype_train = np.float32


class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, run_type, cfg, img_root, ref_root, transform=None):      
        self.cfg = cfg
        self.run_type = run_type
        self.img_root = img_root
        self.ref_root = ref_root
        patch_resolution = cfg.DATASETS.PATCHES_DIR.split('_')[-1]
                        
        if self.run_type not in self.cfg.MODEL.RUN_TYPES: 
            raise ValueError("Invalid run type. Expected one of: %s" % self.cfg.MODEL.RUN_TYPES)
        
        images = sorted(os.listdir(img_root))
        images_df = pd.DataFrame({'image_id':images})
        print(f'all images: {len(images_df)}')  
        
        patches = "ps_" + patch_resolution
        sub_tag = ""
        if "sub" in self.cfg.DATASETS.DATA_DIR:
            sub_tag = "sub2" if "sub2" in self.cfg.DATASETS.DATA_DIR else "sub"
            sub_prefix = f"{sub_tag}_"
        else: sub_prefix = ""
        
        #TODO: fix
        base = "reference_cad_aerial_25_"
        # base = "reference_cad_aerial_8_"

        if cfg.DATASETS.REF_FNAME:
            fname_brk = cfg.DATASETS.REF_FNAME
        else:
            if cfg.MODEL.USE_BRK:
                fname_brk = f"{base}{sub_prefix}{patches}_brk.gpkg"

        if cfg.MODEL.RUN_TYPE == 'predict':  
            tile_dir = os.path.join(cfg.DATASETS.DATA_DIR, 'test', cfg.DATASETS.IMAGES_DIR)
            tile_raster_open = rio.open(os.path.join(tile_dir, f'{self.cfg.MODEL.MOSAIC_TILE[0]}.tif'))
            tile_extents = tile_raster_open.bounds
            clip_mask = box(*tile_extents)
            if cfg.MODEL.USE_BRK: self.brk_reference = gp.read_file(os.path.join(self.ref_root, fname_brk), engine="pyogrio", bbox=clip_mask)
        else:
            if cfg.MODEL.USE_BRK: self.brk_reference = gp.read_file(os.path.join(self.ref_root, fname_brk), engine="pyogrio")
        
        if cfg.MODEL.USE_BRK: print(f'all brk_reference: {len(self.brk_reference)}')      
        
        if self.cfg.DATASETS.USE_CLUSTERS:
            cluster_fname = self.cfg.DATASETS.CLUSTER_FNAME if self.cfg.DATASETS.CLUSTER_FNAME else f'urban_{patch_resolution}_clusters.gpkg'
            cluster_gdf = gp.read_file(os.path.join(self.ref_root, cluster_fname), engine="pyogrio")
            drop_cols = [c for c in ['geometry', 'urban', 'split', 'cluster_type'] if c in cluster_gdf.columns]
            self.subset_patches_ds2 = cluster_gdf.drop(columns=drop_cols)
            if cfg.MODEL.USE_BRK: self.brk_reference = self.brk_reference.merge(self.subset_patches_ds2, how='inner', on='image_id')
            images_df = images_df.merge(self.subset_patches_ds2, how='inner', on='image_id')
            print(f'images after cluster filter: {len(images_df)}')
        else:
            self.subset_patches_ds2 = None
        if cfg.MODEL.RUN_TYPE == 'predict':  
            images_df = images_df[images_df['image_id'].apply(lambda x: any(tile.replace('tile-', '') in x for tile in self.cfg.MODEL.MOSAIC_TILE))]
            print(f'images after sub2 pred tile filter: {len(images_df)}')
            if 'cluster_id' in images_df.columns:
                images_df = images_df[images_df['cluster_id'].apply(lambda x: int(x.strip('"\'')) == int(self.cfg.MODEL.MOSAIC_CLUSTER))].drop(columns=['cluster_id'])
                print(f'images after clustering filter: {len(images_df)}')
      
        if self.cfg.DATASETS.SELECT_VISIBLE and self.cfg.DATASETS.SELECT_INVISIBLE:
            pass
        elif self.cfg.DATASETS.SELECT_VISIBLE:
            if cfg.MODEL.USE_BRK: 
                self.brk_reference = self.brk_reference[self.brk_reference['visible']]
                print(f'visible brk_reference: {len(self.brk_reference)}')
        elif self.cfg.DATASETS.SELECT_INVISIBLE:
            if cfg.MODEL.USE_BRK: 
                self.brk_reference = self.brk_reference[~self.brk_reference['visible']]
                print(f'invisible brk_reference: {len(self.brk_reference)}')
        
        if self.cfg.DATASETS.SELECT_RURAL or self.cfg.DATASETS.SELECT_URBAN:
            urban = gp.read_file(os.path.join(self.ref_root, f'urban_{patch_resolution}.gpkg'), engine='pyogrio')

            sel = []
            if self.cfg.DATASETS.SELECT_RURAL: sel.append(urban[urban['urban'] == False])
            if self.cfg.DATASETS.SELECT_URBAN: sel.append(urban[urban['urban'] == True])
            images_sel = pd.concat(sel)[['image_id']].drop_duplicates().reset_index(drop=True)
            images_df = images_df.merge(images_sel, on='image_id', how='inner')
            print(f'all images after rural/urban filtering: {len(images_df)}')
            
        if cfg.DATASETS.TRAIN_SAMPLE and cfg.MODEL.RUN_TYPE == 'train':
            images_df = images_df.sample(frac=cfg.DATASETS.TRAIN_SAMPLE_RATE, random_state=42)
            print(f'images after train sample filtering: {len(images_df)}')

        if cfg.DATASETS.TEST_SAMPLE and cfg.MODEL.RUN_TYPE == 'predict':
            images_df = images_df.sample(frac=cfg.DATASETS.TEST_SAMPLE_RATE, random_state=42)
            print(f'images after test sample filtering: {len(images_df)}')
                        
        # Only keep patches that have at least one reference polygon (train/val only).
        # During predict (RUN_TYPE != 'train') all patches are used so the full
        # mosaic can be assembled; areas without reference get an all-zero GT.
        if cfg.MODEL.RUN_TYPE == 'train':
            if cfg.MODEL.USE_BRK: images_df = images_df[images_df['image_id'].isin(self.brk_reference['image_id'])]
            print(f'images after reference present filtering: {len(images_df)}')
            
        self.images = images_df['image_id'].tolist()  

        if cfg.MODEL.USE_BRK: 
            self.brk_reference = self.brk_reference[self.brk_reference['image_id'].isin(self.images)]
            print(f'total brk reference after image filtering: {len(self.brk_reference)}')
        
        print(f'total images: {len(images_df)}')
        if cfg.MODEL.USE_BRK: print(f'total brk reference: {len(self.brk_reference)}')
        
        print('---' * 5)
        self.transform = transform
                
    def __getitem__(self, idx_):
       return idx_


class CustomDatasetCad(CustomDataSet):
    def __init__(self, run_type, cfg, img_root, ref_root, transform=None):
        super().__init__(run_type, cfg, img_root, ref_root, transform)    

    def _prepare_geometries(self, gdf, clip_mask):
        geom_series = gdf['geometry']
        if self.cfg.DATASETS.BUFFER != 0: geom_series = geom_series.buffer(self.cfg.DATASETS.BUFFER)
        geom_series = geom_series.clip(mask=clip_mask)
        
        return [shapes for shapes in geom_series]

    def create_mask(self, ann_ids, resolution, clip_mask, transform):
        geom_buffer = self._prepare_geometries(ann_ids.copy(), clip_mask)
        if len(geom_buffer) == 0: return np.zeros([resolution, resolution]).astype(dtype_train)
        seg_mask_buffer = rasterize(shapes=geom_buffer, out_shape=(resolution, resolution), transform=transform, all_touched=True)                         
        
        return seg_mask_buffer.astype(dtype_train)

    def create_visibility_mask(self, ann_ids, resolution, clip_mask, transform):
        ann_ids_copy = ann_ids.copy()
        if 'visible' in ann_ids_copy.columns:
            geom_vis = ann_ids_copy[ann_ids_copy['visible'] == True]
            geom_invis = ann_ids_copy[ann_ids_copy['visible'] == False]
        else:
            geom_vis = ann_ids_copy
            geom_invis = ann_ids_copy.iloc[0:0]
        geom_vis = self._prepare_geometries(geom_vis, clip_mask)
        geom_invis = self._prepare_geometries(geom_invis, clip_mask)
    
        if len(geom_vis) == 0: seg_mask_vis = np.zeros([resolution, resolution])
        else: seg_mask_vis = rasterize(shapes=geom_vis, out_shape=(resolution, resolution), transform=transform, all_touched=True)
        if len(geom_invis) == 0: seg_mask_invis = np.zeros([resolution, resolution])
        else: seg_mask_invis = rasterize(shapes=geom_invis, out_shape=(resolution, resolution), transform=transform, all_touched=True)
        
        stacked_vis_invis = np.stack([seg_mask_vis.astype(dtype_train), seg_mask_invis.astype(dtype_train)], axis=0)

        return stacked_vis_invis.astype(dtype_train)
        
    def create_connectivity_cube(self, seg_mask, resolution):
        img_pad_d1 = np.zeros([resolution + 4, resolution + 4])
        img_pad_d1[2:-2, 2:-2] = seg_mask
        conn_cube_d1 = np.zeros([8, resolution, resolution])

        for i in range(resolution):
            for j in range(resolution):
                if seg_mask[i, j] == 0:
                    continue
                
                conn_cube_d1[0, i, j] = img_pad_d1[i, j + 2]
                conn_cube_d1[1, i, j] = img_pad_d1[i, j - 2]
                conn_cube_d1[2, i, j] = img_pad_d1[i + 2, j]
                conn_cube_d1[3, i, j] = img_pad_d1[i + 2, j + 2]
                conn_cube_d1[4, i, j] = img_pad_d1[i + 2, j - 2]
                conn_cube_d1[5, i, j] = img_pad_d1[i - 2, j]
                conn_cube_d1[6, i, j] = img_pad_d1[i - 2, j + 2]
                conn_cube_d1[7, i, j] = img_pad_d1[i - 2, j - 2]
                         
        img_pad_d3 = np.zeros([resolution + 8, resolution + 8])
        img_pad_d3[4:-4, 4:-4] = seg_mask
        conn_cube_d3 = np.zeros([8, resolution, resolution])

        for i in range(resolution):
            for j in range(resolution):
                if seg_mask[i, j] == 0:
                    continue
                
                conn_cube_d3[0, i, j] = img_pad_d3[i, j + 4]
                conn_cube_d3[1, i, j] = img_pad_d3[i, j - 4]
                conn_cube_d3[2, i, j] = img_pad_d3[i + 4, j]
                conn_cube_d3[3, i, j] = img_pad_d3[i + 4, j + 4]
                conn_cube_d3[4, i, j] = img_pad_d3[i + 4, j - 4]
                conn_cube_d3[5, i, j] = img_pad_d3[i - 4, j]
                conn_cube_d3[6, i, j] = img_pad_d3[i - 4, j + 4]
                
        return conn_cube_d1.astype(dtype_train), conn_cube_d3.astype(dtype_train)
    
    def __getitem__(self, idx_):
        resolution = self.cfg.DATASETS.IMG_SIZE
        img_id = self.images[idx_]
        filename = img_id.replace('.tif', '')
        image = io.imread(os.path.join(self.img_root, img_id)).astype(dtype_train)[:, :, :3] / 255.0
        img_bounds = rio.open(os.path.join(self.img_root, img_id), crs=f'epsg:{self.cfg.DATASETS.CRS}')
    
        scales = {1: "512", 2: "256", 4: "128", 8: "64", 16: "32"}
        ann = {}
        for factor, suffix in scales.items():
            size = resolution // factor

            if self.cfg.MODEL.USE_BRK:
                ann[f'gt_bin_brk_{suffix}_visibility'] = np.zeros((2, size, size)).astype(dtype_train)
                ann[f'gt_bin_brk_{suffix}'] = np.zeros((size, size)).astype(dtype_train)
                
                ann[f'gt_cc_d1_brk_{suffix}_vis_'] = np.zeros((8, size, size)).astype(dtype_train)
                ann[f'gt_cc_d3_brk_{suffix}_vis_'] = np.zeros((8, size, size)).astype(dtype_train)
                ann[f'gt_cc_d1_brk_{suffix}_inv_'] = np.zeros((8, size, size)).astype(dtype_train)
                ann[f'gt_cc_d3_brk_{suffix}_inv_'] = np.zeros((8, size, size)).astype(dtype_train)
                
                ann[f'gt_cc_d1_brk_{suffix}'] = np.zeros((8, size, size)).astype(dtype_train)
                ann[f'gt_cc_d3_brk_{suffix}'] = np.zeros((8, size, size)).astype(dtype_train)
                
        if self.cfg.MODEL.USE_BRK:
            brk_ann_ids = self.brk_reference[self.brk_reference['image_id'] == img_id]
            if len(brk_ann_ids) > 0:  
                for res in self.cfg.DATASETS.RESOLUTIONS:  
                    clip_mask = box(*img_bounds.bounds) # https://gis.stackexchange.com/questions/352445/make-shapefile-from-raster-bounds-in-python
                    transform = rio.transform.from_bounds(*img_bounds.bounds, width=res, height=res)
                    
                    brk_mask = self.create_mask(brk_ann_ids, res, clip_mask, transform)
                    ann[f'gt_bin_brk_{res}'] = brk_mask
                    
                    brk_connect = self.create_connectivity_cube(brk_mask, res)
                    ann[f'gt_cc_d1_brk_{res}'] = brk_connect[0]
                    ann[f'gt_cc_d3_brk_{res}'] = brk_connect[1]   
                    
                    brk_vis_mask = self.create_visibility_mask(brk_ann_ids, res, clip_mask, transform)
                    ann[f'gt_bin_brk_{res}_visibility'] = brk_vis_mask

                    brk_connect_vis = self.create_connectivity_cube(brk_vis_mask[0,:,:], res)
                    ann[f'gt_cc_d1_brk_{res}_vis_'] = brk_connect_vis[0]
                    ann[f'gt_cc_d3_brk_{res}_vis_'] = brk_connect_vis[1]   
                    
                    brk_connect_inv = self.create_connectivity_cube(brk_vis_mask[1,:,:], res)
                    ann[f'gt_cc_d1_brk_{res}_inv_'] = brk_connect_inv[0]
                    ann[f'gt_cc_d3_brk_{res}_inv_'] = brk_connect_inv[1]
                   
        if self.transform is not None:
            return self.transform(image, ann, filename)
        
        return image, ann, filename

    def __len__(self):
        return len(self.images)

def collate_fn(batch):
    return (default_collate([b[0] for b in batch]),
            default_collate([b[1] for b in batch]), 
            [b[2] for b in batch]
        )

    