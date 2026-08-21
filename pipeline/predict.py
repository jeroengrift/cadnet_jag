import numpy as np
import skan
import skimage
import skimage.morphology
import shapely.geometry
import os 
import geopandas as gp
import pandas as pd
import csv
import sys
import math
import networkx as nx
import rasterio
import torch
import shapely
import torch.nn.functional as F
from affine import Affine
from rasterio.features import rasterize

from rasterio.crs import CRS
from shapely.geometry import Point, LineString, MultiPoint, box
from shapely import line_merge
from shapely.ops import split, substring
from torchview import draw_graph
from rtree import index
from rasterio.warp import reproject, Resampling

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.comm import to_single_device
from typing import List
from pyEdgeEval._lib import correspond_pixels


class Paths:
    def __init__(self, indices=None, indptr=None):
        if indices is None:
            self.indices = np.empty(0, dtype=np.longlong)
        else:
            self.indices = indices
        if indptr is None:
            self.indptr = np.empty(0, dtype=np.longlong)
        else:
            self.indptr = indptr


class Skeleton:
    def __init__(self, coordinates=None, paths=None, degrees=None):
        if coordinates is None:
            self.coordinates = np.empty((0, 2), dtype=np.float64)
        else:
            self.coordinates = coordinates
        if paths is None:
            self.paths = Paths()
        else:
            self.paths = paths
        if degrees is None:
            self.degrees = np.empty(0, dtype=np.longlong)
        else:
            self.degrees = degrees
# from apls.apls import run_apls

pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199


# https://stackoverflow.com/questions/62990029/how-to-get-equally-spaced-points-on-a-line-in-shapely/62994304#62994304
def create_segments(line, max_segment_length):
    mp = shapely.geometry.MultiPoint()
    for i in np.arange(0, line.length, max_segment_length):
        s = substring(line, i, i+max_segment_length)
        mp = mp.union(s.boundary)
    return mp    

def segment_geometries(geo_df, max_seg_length=10):    
    segmented_geoms = []
    for idx, row in geo_df.iterrows():
        line = row.geometry
        seg_points = create_segments(line, max_seg_length)
        seg_points = [p for p in seg_points.geoms]
        seg_points = sorted(seg_points, key=lambda p: line.project(p))
        for start, end in zip(seg_points, seg_points[1:]):
            segmented_geoms.append(shapely.geometry.LineString([start, end]))
    return segmented_geoms 

def extract_endpoints(geom):
    if geom.geom_type == 'LineString' and len(geom.coords) >= 2:
        return shapely.geometry.LineString([geom.coords[0], geom.coords[-1]])
    else: 
        return geom

def calculate_azimuth(geo_df):
    azimuth_list = []
    geo_df = geo_df[geo_df.geometry.type == 'LineString']

    for i, row in enumerate(geo_df.itertuples()):
        if isinstance(row.geometry, LineString):
            point_list = list(row.geometry.coords)
            x0, x1 = point_list[0][0], point_list[1][0]
            y0, y1 = point_list[0][1], point_list[1][1]
            azimuth = math.atan2((x1 - x0), (y1 - y0))

            if azimuth > math.pi:
                azimuth -= math.pi
            elif 0 > azimuth > -math.pi:
                azimuth += math.pi
            elif -math.pi > azimuth > -(2 * math.pi):
                azimuth += (2 * math.pi)
            azimuth_list.append(azimuth)

    geo_df['azimuth'] = azimuth_list

    return geo_df

def pair_points(list):
    for i in range(0, len(list) - 1):
        yield list[i-1], list[i]

def chop_lines(gdf): 
    gdf_chopped = {'geometry': []}

    for i in range(len(gdf)):
        geom_column = gdf.geometry[i]
        if isinstance(geom_column, LineString):
            linestring = gdf.geometry[i]
            coordinates = linestring.coords    
        elif isinstance(geom_column, gp.geoseries.GeoSeries):
            linestring = gdf.geometry[i].iloc[0]
            coordinates = list(gdf.geometry[i].iloc[0].coords)
                            
        points = []
        
        for seg_start, seg_end in pair_points(coordinates):
            line_start = Point(seg_start)
            line_end = Point(seg_end) 
            points.append(line_start)
            points.append(line_end)

        chop_points = MultiPoint(points)       
        geom_collection = list(split(linestring, chop_points).geoms)

        for j in range(len(geom_collection)):    
            result = LineString(geom_collection[j])
            gdf_chopped['geometry'].append(result)
        
    return gp.GeoDataFrame.from_dict(gdf_chopped)

def concatenate_lines(gdf, threshold):   
    polylines_azimuths = calculate_azimuth(gdf)
    polylines_azimuths["id"] = polylines_azimuths.index + 1

    polylines_azimuths_nn = gp.sjoin(polylines_azimuths, polylines_azimuths,
        how="inner",
        predicate="intersects",
        lsuffix="left",
        rsuffix="right")
    
    polylines_azimuths_nn = polylines_azimuths_nn[polylines_azimuths_nn.index != polylines_azimuths_nn.index_right]
    polylines_azimuths_nn = polylines_azimuths_nn.drop(columns=['index_right'])

    polylines_azimuths_nn = polylines_azimuths_nn[(polylines_azimuths_nn.azimuth_left <= (polylines_azimuths_nn.azimuth_right + threshold)) 
            & (polylines_azimuths_nn.azimuth_left >= (polylines_azimuths_nn.azimuth_right - threshold))].reset_index().drop(columns=['index'])

    graph_list = [sorted((row.id_left, row.id_right)) for _, row in polylines_azimuths_nn.iterrows()]

    if len(graph_list) > 0:
        connected_components = list(map(sorted, nx.connected_components(nx.Graph(graph_list))))
        connected_components = list([i, def_instance] for i, def_instances in enumerate(connected_components)
                for def_instance in def_instances)
        
        connected_components = pd.DataFrame(connected_components).rename(columns={0: 'graph_id', 1: 'seg_id'})
        connected_components['graph'] = True
        connected_components = gp.GeoDataFrame(polylines_azimuths.merge(connected_components, left_on='id', right_on='seg_id', how='left'), geometry='geometry')
        connected_components.fillna({'graph': False}, inplace=True)
        
        out_graph = connected_components[~connected_components['graph']].reset_index().drop(columns=['index'])
        in_graph = connected_components[connected_components['graph']].reset_index().drop(columns=['index'])

        agg_segments = in_graph.dissolve(by='graph_id').drop(columns=['seg_id', 'azimuth', 'id', 'graph'], axis=1).reset_index()
        agg_segments['geometry'] = agg_segments['geometry'].apply(line_merge)
        agg_segments_list = pd.DataFrame(in_graph.groupby('graph_id')['seg_id'].apply(list))
        agg_segments = agg_segments.merge(agg_segments_list, on='graph_id', how='left')

        merged_segments = pd.concat([agg_segments, out_graph], sort=True) \
            .drop(columns=['id', 'azimuth'], axis=1).reset_index() \
            .drop(columns=['index'])
        
        merged_segments = merged_segments.reset_index().drop(columns=['index'])
        
        return merged_segments

    else:
        return gdf

def compare_polys(poly_a, poly_b, index):
    dist = polis(poly_a.coords, poly_b)
    dist += polis(poly_b.coords, poly_a)
    return (dist, index)

def polis(coords, boundary):
    summ = 0.
    for pt in (Point(c) for c in coords):
        summ += boundary.distance(pt)
    return summ / float(len(coords))

def calculate_polis(reference_gdf, vector_gdf):    
    ref_polys_id = reference_gdf['id'].to_list()
    ref_polys = reference_gdf['geometry'].to_list()

    idx = index.Index((i, geom.bounds, None) for i, geom in enumerate(ref_polys))

    vector_gdf['polis'] = None    
    vector_gdf['match_id'] = None

    for index_, row_ in vector_gdf.iterrows():
        row_geom = vector_gdf.loc[index_, 'geometry']
        ref_pindices = [i for i in idx.nearest(row_geom.bounds, 100)]        
        scores = [compare_polys(row_geom, ref_polys[i], i) for i in ref_pindices]
        polis_score = min(scores, key = lambda t: t[0])                
        vector_gdf.loc[index_, 'polis', ] = polis_score[0]
        vector_gdf.loc[index_, 'match_id', ] = ref_polys_id[polis_score[1]]

    polis_score = vector_gdf['polis'].sum()

    return vector_gdf, polis_score

def gaussian_kernel(x_size, y_size=None, mu=0.0, sigma=0.5, normalize=True):
    if y_size is None:
        y_size = x_size
    x = np.linspace(-1, 1, x_size)
    y = np.linspace(-1, 1, y_size)
    xv, yv = np.meshgrid(x, y)
    r2 = xv**2 + yv**2
    # 2D Gaussian (centered); if mu != 0 you'd shift radius, but typically mu=0
    kernel = np.exp(- (r2) / (2 * (sigma**2)))
    if normalize:
        kernel = kernel / kernel.sum()
    return kernel


class BaseVectorizer:
    def __init__(self, cfg, file_output_dir):
        self.cfg = cfg
        self.file_output_dir = file_output_dir

    def skeleton_to_polylines(self, skeleton: Skeleton) -> List[np.ndarray]:
        polylines = []
        for path_i in range(skeleton.paths.indptr.shape[0] - 1):
            start, stop = skeleton.paths.indptr[path_i:path_i + 2]
            path_indices = skeleton.paths.indices[start:stop]
            path_coordinates = skeleton.coordinates[path_indices]
            polylines.append(path_coordinates)
        return polylines

    def compute_skeletons(self, seg_batch, cfg) -> List[Skeleton]:
        assert len(seg_batch.shape) == 4 and seg_batch.shape[1] <= 3, "seg_batch should be (N, C, H, W) with C <= 3, not {}".format(seg_batch.shape)

        corrected_edge_mask_batch = cfg.DATASETS.BINARY_TRESHOLD < seg_batch[:, 0, :, :]
        np_corrected_edge_mask_batch = corrected_edge_mask_batch
        np_corrected_edge_mask_batch = np_corrected_edge_mask_batch.squeeze() 
        np_edge_mask_padded = np.pad(np_corrected_edge_mask_batch, pad_width=2, mode="edge")
        
        ## Morphological closing on an image is defined as a dilation followed by an erosion. Closing can remove small dark spots and connect small bright cracks.
        np_edge_mask_padded = skimage.morphology.binary_closing(np_edge_mask_padded)
        ## skeletonize works by making successive passes of the image. On each pass, border pixels are identified and removed on the condition that they do not break the connectivity of the corresponding object.
        skeleton_image = skimage.morphology.skeletonize(np_edge_mask_padded)
        # skeleton_image, dist = skimage.morphology.medial_axis(np_edge_mask_padded, return_distance=True)
        skeleton_image = skeleton_image[2:-2, 2:-2]

        # sk_image = torch.from_numpy(skeleton_image).unsqueeze(0).unsqueeze(0)
        # # plot skeleton images, dir exist
        # try:
        #     plot_features(sk_image, os.path.join(self.file_output_dir, f'skeleton_{self.type_}.jpeg'))
        # except:
        #     pass
        
        skeleton = Skeleton()
        if 0 < skeleton_image.sum():
            try:
                # skan does not work in some cases (paths of 2 pixels or less, etc) which raises a ValueError, in witch case we continue with an empty skeleton
                # What is the end result of skan? It creates a network from a skeleton image
                # 
                skeleton = skan.Skeleton(skeleton_image, keep_images=False)            
                skeleton.coordinates = skeleton.coordinates[:skeleton.paths.indices.max() + 1]
                if skeleton.coordinates.shape[0] != skeleton.degrees.shape[0]:
                    raise ValueError(f"skeleton.coordinates.shape[0] = {skeleton.coordinates.shape[0]} while skeleton.degrees.shape[0] = {skeleton.degrees.shape[0]}. They should be of same size.")
            except ValueError as e:
                print(e)
            
            # # not always existing dir
            # try:
            #     polylines_coords = self.skeleton_to_polylines(skeleton)
            #     plt.imshow(skeleton_image)
            #     for polyline_coords in polylines_coords:
            #         plt.plot(polyline_coords[:, 1], polyline_coords[:, 0])
            #     plt.show()
            #     plt.savefig(os.path.join(self.file_output_dir, f'skeleton_polylines_{self.type_}.jpeg'), bbox_inches='tight')
            # except:
            #     pass
                
        return skeleton


class VectorizerSimple(BaseVectorizer):
    def __init__(self, cfg, file_output_dir):
        super().__init__(cfg, file_output_dir)

    def __call__(self, seg_batch):
        assert len(seg_batch.shape) == 4 and seg_batch.shape[1] <= 3, "seg_batch should be (N, C, H, W) with C <= 3, not {}".format(seg_batch.shape)

        skeletons_batch = self.compute_skeletons(seg_batch, self.cfg)
        polylines_batch = [self.skeleton_to_polylines(skeleton) for skeleton in [skeletons_batch]]
        
        return polylines_batch
    
    
class PredictRaster():
    def __init__(self, cfg, output_dir, model=None, test_data=None):
        self.cfg = cfg
        self.model = model
        self.test_data = test_data
        self.output_dir = output_dir    
        self.use_vis = ''
        self.vis_types = {'': 0}
        self.output_mosaic_dir = os.path.normpath(os.path.join(self.output_dir, self.cfg.DATASETS.MOSAIC_DIR))
        self.tile_dir = os.path.join(cfg.DATASETS.DATA_DIR, 'test', cfg.DATASETS.IMAGES_DIR)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.tile_size = cfg.DATASETS.TILE_SIZE
        self.patch_resolution = cfg.DATASETS.PATCHES_DIR.split('_')[-1]
        self.max_distance_raster = cfg.MODEL.MAX_DIST

        patches = "ps_" + self.patch_resolution
        dataset_name = os.path.basename(self.cfg.DATASETS.DATA_DIR)
        self.ref_root = os.path.join(self.cfg.DATASETS.DATA_DIR, self.cfg.DATASETS.REF_DIR, '')

        base = '' if not self.cfg.MODEL.TEST_OVERLAP_50 else "overlap_50/"
        if self.cfg.DATASETS.REF_FNAME:
            if self.cfg.MODEL.TEST_OVERLAP_50:
                overlap_ref = os.path.join(self.ref_root, 'overlap_50', self.cfg.DATASETS.REF_FNAME)
                self.fname = os.path.join('overlap_50', self.cfg.DATASETS.REF_FNAME) if os.path.exists(overlap_ref) else self.cfg.DATASETS.REF_FNAME
            else:
                self.fname = self.cfg.DATASETS.REF_FNAME
        elif cfg.MODEL.USE_BRK:
            self.fname = f"{base}reference_{dataset_name}_{patches}_brk.gpkg"

        if self.cfg.DATASETS.USE_URBAN_RURAL:
            urban_rural_file = self.ref_root + f'{base}urban_{self.patch_resolution}.gpkg'
            self.images_urban_rural = gp.read_file(urban_rural_file)
        else:
            self.images_urban_rural = None

        if self.cfg.DATASETS.USE_CLUSTERS:
            cluster_fname = self.cfg.DATASETS.CLUSTER_FNAME if self.cfg.DATASETS.CLUSTER_FNAME else f'{base}urban_{self.patch_resolution}_clusters.gpkg'
            cluster_file = self.ref_root + cluster_fname
            self.clusters = gp.read_file(cluster_file).set_crs(f'EPSG:{self.cfg.DATASETS.CRS}')
            if 'cluster_id' not in self.clusters.columns:
                self.clusters['cluster_id'] = '0'
        else:
            self.clusters = None

    def write(self):   
        tile = self.cfg.MODEL.MOSAIC_TILE[0]
       
        if self.cfg.MODEL.USE_BRK: gt_type = 'brk'
        if os.path.isdir(self.output_mosaic_dir): pass
        else: os.makedirs(self.output_mosaic_dir, exist_ok=True) 

        if self.cfg.DATASETS.USE_CLUSTERS:
            tile_clusters = self.clusters[self.clusters['image_id'].str.contains(tile, na=False)]
            tile_clusters_dissolved = gp.GeoDataFrame(tile_clusters.dissolve(by='cluster_id')).reset_index()
        else:
            with rasterio.open(os.path.join(self.tile_dir, f'{tile}.tif')) as src:
                tile_bounds = src.bounds
            tile_geom = box(*tile_bounds)
            tile_clusters_dissolved = gp.GeoDataFrame([{'cluster_id': '0', 'geometry': tile_geom}], crs=f'EPSG:{self.cfg.DATASETS.CRS}')
            tile_clusters = tile_clusters_dissolved.copy()
            tile_clusters['image_id'] = tile
        
        print(f'Processing tile {tile} with {len(tile_clusters_dissolved)} clusters')
        with rasterio.open(os.path.join(self.tile_dir, f'{tile}.tif')) as src_tile:
            tile_bounds = src_tile.bounds
            # derive scale from actual tile pixel count; avoids relying on DOWNSAMPLE_FACTOR config
            actual_downsample = self.tile_size / src_tile.width
            tile_transform = rasterio.transform.from_bounds(*tile_bounds, width=self.tile_size, height=self.tile_size)
        
        for _, cluster_row in tile_clusters_dissolved.iterrows():
            cluster_id = cluster_row['cluster_id']
                    
            if cluster_id != str(self.cfg.MODEL.MOSAIC_CLUSTER):
                continue
            
            print(f'Processing cluster {cluster_id} in tile {tile}')
            cluster_geom = cluster_row.geometry                
            cluster = tile_clusters[tile_clusters['cluster_id'] == cluster_id]                            

            cluster_minx, cluster_miny, cluster_maxx, cluster_maxy = cluster_geom.bounds
            cluster_width = cluster_maxx - cluster_minx
            cluster_height = cluster_maxy - cluster_miny

            tile_w = int(math.ceil(cluster_width / (float(self.patch_resolution)/100)))     
            tile_h = int(math.ceil(cluster_height / (float(self.patch_resolution)/100)))           

            if self.cfg.MODEL.WRITE_MOSAIC_RASTER:
                mosaics = {}

                cluster_extents = cluster_geom.bounds
                cluster_transform = rasterio.transform.from_bounds(*cluster_extents, width=tile_w, height=tile_h)
                clip_mask = box(*cluster_extents)                    

                gt_polylines = gp.read_file(os.path.join(self.ref_root, self.fname), engine="pyogrio", bbox=clip_mask)
                if 'filename_10k' not in gt_polylines.columns and 'image_id' in gt_polylines.columns:
                    gt_polylines['filename_10k'] = gt_polylines['image_id'].str.extract(r'((?:tile-)?\d+-\d+)')
                if 'filename_10k' in gt_polylines.columns:
                    tile_numeric = tile.removeprefix('tile-')
                    gt_polylines = gt_polylines[gt_polylines['filename_10k'].str.contains(tile_numeric, na=False, regex=False)]
                                                
                def create_layer_dict():
                    base = {}
                    fp_tensor = torch.full((tile_h, tile_w), float('nan'), dtype=torch.float32)
                    base[''] = fp_tensor
                    count_tensor = torch.zeros((tile_h, tile_w), dtype=torch.float32)
                    base['count_array'] = count_tensor
                    return base

                mosaics[f'{gt_type}'] = create_layer_dict()

                if self.cfg.DATASETS.BUFFER > 0: _gt_ = gt_polylines['geometry'].buffer(self.cfg.DATASETS.BUFFER).clip(mask=clip_mask)
                else: _gt_ = gt_polylines['geometry'].clip(mask=clip_mask)

                _gt_shapes = [geom for geom in _gt_.geometry]
                if _gt_shapes:
                    _gt_arr = rasterize(shapes=_gt_shapes,
                                        out_shape=(tile_h, tile_w),
                                        transform=cluster_transform,
                                        all_touched=True)
                else:
                    _gt_arr = np.zeros((tile_h, tile_w), dtype=np.uint8)
                mosaics[f'{gt_type}']['_gt_'] = torch.from_numpy(_gt_arr).to(torch.uint8)
                
                if 'visible' in gt_polylines.columns:
                    if self.cfg.DATASETS.BUFFER > 0:
                        _gt_vis = gt_polylines[gt_polylines['visible'] == True]['geometry'].buffer(self.cfg.DATASETS.BUFFER).clip(mask=clip_mask)
                    else:
                        _gt_vis = gt_polylines[gt_polylines['visible'] == True]['geometry'].clip(mask=clip_mask)
                   
                    _gt_vis_shapes = [geom for geom in _gt_vis.geometry]
                    if _gt_vis_shapes:
                        _gt_vis_arr = rasterize(shapes=_gt_vis_shapes,
                                                out_shape=(tile_h, tile_w),
                                                transform=cluster_transform,
                                                all_touched=True)
                    else:
                        _gt_vis_arr = np.zeros((tile_h, tile_w), dtype=np.uint8)
                    mosaics[f'{gt_type}']['_gt__vis_'] = torch.from_numpy(_gt_vis_arr).to(torch.uint8)

                    if self.cfg.DATASETS.BUFFER > 0:
                        _gt_inv = gt_polylines[gt_polylines['visible'] == False]['geometry'] \
                            .buffer(self.cfg.DATASETS.BUFFER) \
                            .clip(mask=clip_mask)
                    else:
                        _gt_inv = gt_polylines[gt_polylines['visible'] == False]['geometry'].clip(mask=clip_mask)
                    
                    _gt_inv_shapes = [geom for geom in _gt_inv.geometry]
                    if len(_gt_inv_shapes):
                        _gt_inv_arr = rasterize(shapes=_gt_inv_shapes,
                                                out_shape=(tile_h, tile_w),
                                                transform=cluster_transform,
                                                all_touched=True)
                        mosaics[f'{gt_type}']['_gt__inv_'] = torch.from_numpy(_gt_inv_arr).to(torch.uint8)
                                    
            def iterate_data():
                cluster_filenames = set()
                for val in cluster['image_id'].astype(str).values:
                    cluster_filenames.add(os.path.splitext(val)[0])

                for it, (test_images, test_annotations, test_filenames) in enumerate(self.test_data):
                    filename = test_filenames[0]
                    fname_no_ext = os.path.splitext(filename)[0]
                                                                                                    
                    if self.cfg.MODEL.WRITE_MOSAIC_RASTER and self.cfg.DATASETS.USE_CLUSTERS:
                        if fname_no_ext not in cluster_filenames:
                            continue
                        
                    print(f'Processing {filename} in cluster {cluster_id}')                            
                    test_images = test_images.to(self.device)
                    test_annotations = to_single_device(test_annotations, self.device)
                        
                    with torch.no_grad():
                        torch.cuda.set_device(int(self.cfg.MODEL.DEVICE.split(':')[-1]))
                        autocast_device = self.cfg.MODEL.DEVICE.split(':')[0] if isinstance(self.cfg.MODEL.DEVICE, str) and 'cuda' in self.cfg.MODEL.DEVICE else self.cfg.MODEL.DEVICE
                        with torch.autocast(device_type=autocast_device, dtype=torch.float16):
                            preds = self.model(test_images, test_annotations)

                    if it == 0:
                        model_graph = draw_graph(self.model, input_data=(test_images, test_annotations)).visual_graph
                        graph_svg = model_graph.pipe(format='png')
                        save_path = os.path.join(self.output_dir, 'model_graph.png')
                        with open(save_path, 'wb') as f:
                            f.write(graph_svg)
                    
                    def process_raster(type_, it):     
                        if self.cfg.MODEL.USE_MULTI:
                            resolutions = [f's{res}' for res in self.cfg.DATASETS.RESOLUTIONS[::-1]]
                            resolutions.append(512)
                        else:
                            resolutions = [512]
                            
                        if self.cfg.MODEL.BACKBONE.startswith('unet'):
                            resolutions = [i for i in resolutions if i != 's32']
                        
                        if 'visible' in gt_polylines.columns:
                            self.use_vis = '_visibility'
                            self.vis_types = {'_vis_' : 0, '_inv_' : 1}
                        else:
                            self.use_vis = ''
                            self.vis_types = {'': 0}
                                                        
                        for resolution in resolutions:
                            if resolution == 512: 
                                pass
                            else: 
                                continue
                                                            
                            pred_bin_edge = preds[f"pred_bin_{type_}_{resolution}"].squeeze().unsqueeze(0).unsqueeze(0)
                            pred_grad_edge = pred_bin_edge.clone()  
                            pred_bin_edge[pred_bin_edge >= self.cfg.DATASETS.BINARY_TRESHOLD] = 1.0
                            pred_bin_edge[pred_bin_edge < self.cfg.DATASETS.BINARY_TRESHOLD] = 0.0  

                            if self.cfg.MODEL.USE_COA:
                                pred_connect_d1 = preds[f'pred_cc_d1_{type_}_{resolution}']
                                pred_connect_d3 = preds[f'pred_cc_d3_{type_}_{resolution}']
                                
                                pred_connect_d1_grad = pred_connect_d1.clone()
                                pred_connect_d1[pred_connect_d1 >= self.cfg.DATASETS.BINARY_TRESHOLD] = 1.0
                                pred_connect_d1[pred_connect_d1 < self.cfg.DATASETS.BINARY_TRESHOLD] = 0.0
                                
                                pred_connect_d3_grad = pred_connect_d3.clone()
                                pred_connect_d3[pred_connect_d3 >= self.cfg.DATASETS.BINARY_TRESHOLD] = 1.0
                                pred_connect_d3[pred_connect_d3 < self.cfg.DATASETS.BINARY_TRESHOLD] = 0.0
                                
                                pred_bin_edge = torch.cat((pred_bin_edge, pred_connect_d1, pred_connect_d3), dim=1)
                                pred_bin_edge = torch.amax(pred_bin_edge, axis=1).unsqueeze(1)
                                    
                                pred_grad_edge = torch.cat((pred_grad_edge, pred_connect_d1_grad, pred_connect_d3_grad), dim=1)
                                pred_grad_edge = torch.amax(pred_grad_edge, axis=1).unsqueeze(1)
                                
                            if self.cfg.MODEL.WRITE_MOSAIC_RASTER and resolution == 512:
                                # this was not very accurate, therefore snap grid
                                parts = filename.split("_")
                                c_off = int(parts[1]) * self.cfg.DATASETS.DOWNSAMPLE_FACTOR
                                r_off = int(parts[2]) * self.cfg.DATASETS.DOWNSAMPLE_FACTOR
                                
                                x_ul, y_ul = rasterio.transform.xy(tile_transform, r_off, c_off, offset='ul')                                    
                                col_f, row_f = (~cluster_transform) * (x_ul, y_ul)                                        
                                cluster_r_start = int(math.floor(row_f))
                                cluster_c_start = int(math.floor(col_f))
                                
                                r_start = max(0, int(np.floor(cluster_r_start)))
                                c_start = max(0, int(np.floor(cluster_c_start)))
                                r_end = min(tile_h, int(np.ceil(cluster_r_start + self.cfg.DATASETS.IMG_SIZE)))
                                c_end = min(tile_w, int(np.ceil(cluster_c_start + self.cfg.DATASETS.IMG_SIZE)))
                                
                                target_block = mosaics[type_][''][r_start:r_end, c_start:c_end]
                                if target_block.shape[0] == 0 or target_block.shape[1] == 0:
                                    continue
                                d_arr = pred_grad_edge.squeeze().to(mosaics[type_][''].device)    
                                
                                if d_arr.shape[:2] != target_block.shape:
                                    d_arr = d_arr.unsqueeze(0).unsqueeze(0)
                                    d_arr = F.interpolate(d_arr, size=target_block.shape, mode='nearest')
                                    d_arr = d_arr.squeeze(0).squeeze(0)
                                        
                                weights = gaussian_kernel(d_arr.shape[1], d_arr.shape[0], sigma=0.5, normalize=True)
                                weights = torch.from_numpy(weights).to(d_arr.device).to(d_arr.dtype)
                                
                                target_block = torch.nan_to_num(target_block, nan=0.0)                                                                                                                
                                mosaics[type_][''][r_start:r_end, c_start:c_end] = target_block + d_arr * weights

                                # ones = torch.ones_like(d_arr, dtype=mosaics[type_]['count_array'].dtype, device=mosaics[type_]['count_array'].device)
                                mosaics[type_]['count_array'][r_start:r_end, c_start:c_end] = torch.add(mosaics[type_]['count_array'][r_start:r_end, c_start:c_end], weights)
                                        
                    process_raster(f'{gt_type}', it)
            
            if self.cfg.MODEL.WRITE_MOSAIC_RASTER:
                iterate_data()
        
            if self.cfg.MODEL.WRITE_MOSAIC_RASTER:
                types = ['brk']
                for type_ in types:
                    if not getattr(self.cfg.MODEL, f'USE_{type_.upper()}'): 
                        continue  
                    
                    mosaics[type_][''] = torch.div(mosaics[type_][''], mosaics[type_]['count_array'])
                        
                    for layer_name, layer_data in mosaics[type_].items():
                        if isinstance(layer_data, torch.Tensor):
                            mosaics[type_][layer_name] = layer_data.detach().cpu().numpy()
                            
                    for vis_type in mosaics[type_]:
                        if '_gt_' not in vis_type and 'count_array' not in vis_type:
                            print(f'Writing mosaic for {tile} {cluster_id}  {type_} {vis_type}')
                            
                            arr = mosaics[type_][vis_type]                           
                            mask = ~np.isnan(arr)
                            rows = np.any(mask, axis=1)
                            cols = np.any(mask, axis=0)
                            if not rows.any() or not cols.any():
                                print(f'Skipping empty mosaic for {tile} {cluster_id} {type_} {vis_type}')
                                continue
                            arr_cropped = arr[np.ix_(rows, cols)]
                            
                            row_start, col_start = np.where(rows)[0][0], np.where(cols)[0][0]
                                                            
                            cropped_transform = Affine(
                                cluster_transform.a, cluster_transform.b, cluster_transform.c + col_start * cluster_transform.a,
                                cluster_transform.d, cluster_transform.e, cluster_transform.f + row_start * cluster_transform.e
                            )
                            
                            out_arr = np.zeros_like(arr, dtype='float32')

                            reproject(
                                source=arr_cropped,
                                destination=out_arr,
                                src_transform=cropped_transform,
                                src_crs=CRS.from_epsg(self.cfg.DATASETS.CRS),
                                dst_transform=cluster_transform,    
                                dst_crs=CRS.from_epsg(self.cfg.DATASETS.CRS),
                                resampling=Resampling.bilinear      
                            )
                            
                            with rasterio.open(os.path.join(self.output_mosaic_dir, f'{tile}_{cluster_id}_{type_}{vis_type}.tif'),
                                            mode='w',
                                            driver='GTiff',
                                            height=out_arr.shape[0],
                                            width=out_arr.shape[1],
                                            transform=cluster_transform,
                                            count=1,
                                            dtype='float32',
                                            crs=CRS.from_epsg(self.cfg.DATASETS.CRS)
                                            ) as dest:
                                dest.write(out_arr, 1)
                                
                        elif '_gt_' in vis_type:    
                            print(f'Writing mosaic for {tile} {cluster_id}  {type_} {vis_type} gt')
                            with rasterio.open(os.path.join(self.output_mosaic_dir, f'{tile}_{cluster_id}_{type_}{vis_type}.tif'),
                                            mode='w',
                                            driver='GTiff',
                                            height=mosaics[type_][vis_type].shape[0],
                                            width=mosaics[type_][vis_type].shape[1],
                                            transform=cluster_transform,
                                            count=1,
                                            dtype='uint8',
                                            crs=CRS.from_epsg(self.cfg.DATASETS.CRS)
                                            ) as dest:
                                dest.write(mosaics[type_][vis_type], 1)
                                dest.close()    

    def evaluate(self):
        def compute_raster_metrics(pred, gt, cfg, tile, cluster, type_, vis_type, u_t, raster_writer, max_dist=5):
            print(f'Computing raster metrics for {tile}, {cluster}, {type_}, {vis_type}, {u_t} with max_dist {max_dist}')

            if pred.max() > 1.0:
                pred = pred / 255.0
            if gt.max() > 1.0:
                gt = gt / 255.0

            valid_pixels = np.count_nonzero(~np.isnan(pred) & ~np.isnan(gt))

            pred = (pred >= cfg.DATASETS.BINARY_TRESHOLD).astype(np.uint8)
            gt = (gt >= cfg.DATASETS.BINARY_TRESHOLD).astype(np.uint8)
            
            # ones_pred = int(np.count_nonzero(pred == 1))
            # zeros_pred = int(np.count_nonzero(pred == 0))
            # ones_gt = int(np.count_nonzero(gt == 1))
            # zeros_gt = int(np.count_nonzero(gt == 0))
            # print(f'pred -> ones: {ones_pred}, zeros: {zeros_pred}')
            # print(f'gt   -> ones: {ones_gt}, zeros: {zeros_gt}')
        
            pred = np.pad(pred, pad_width=2, mode="edge")
            pred = skimage.morphology.binary_closing(pred)
            pred = skimage.morphology.skeletonize(pred)
            pred = pred[2:-2, 2:-2]
            
            # ones_pred = int(np.count_nonzero(pred == 1))
            # zeros_pred = int(np.count_nonzero(pred == 0))
            # print(f'pred -> ones: {ones_pred}, zeros: {zeros_pred}')
              
            # Split data into 512x512 patches (or smaller if at the edge). Function is too slow for large rasters.
            patch_size = 512
            h, w = pred.shape
            o_pred = np.zeros_like(pred, dtype=np.uint8)
            o_gt = np.zeros_like(gt, dtype=np.uint8)

            idiag = math.sqrt(patch_size**2 + patch_size**2)
            max_dist_pixel = max_dist / idiag
            max_dist_pixel = float(max_dist_pixel.real)

            for row in range(0, h, patch_size):
                for col in range(0, w, patch_size):
                    row_end = min(row + patch_size, h)
                    col_end = min(col + patch_size, w)
                    pred_patch = pred[row:row_end, col:col_end]
                    gt_patch = gt[row:row_end, col:col_end]

                    o_pred_patch, o_gt_patch, cost_patch, oc_patch = correspond_pixels(pred_patch, gt_patch, max_dist=max_dist_pixel)

                    o_pred[row:row_end, col:col_end] = o_pred_patch
                    o_gt[row:row_end, col:col_end] = o_gt_patch

            sum_pred = pred.sum()
            sum_gt   = gt.sum()

            tp_pred = o_pred.astype(bool).sum()
            tp_gt   = o_gt.astype(bool).sum()

            precision = tp_pred / sum_pred if sum_pred > 0 else 0
            recall    = tp_gt   / sum_gt   if sum_gt   > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

            tp = tp_pred
            fp = sum_pred - tp_pred
            fn = sum_gt - tp_gt
            tn = valid_pixels - tp - fp - fn
            
            print(f'precision: {precision}, recall: {recall}, f1: {f1}, tp: {tp}, fp: {fp}, fn: {fn}, tn: {tn}')
                                                                                            
            raster_writer.writerow([tile, cluster, type_, u_t, max_dist, tn, fp, fn, tp, precision, recall, f1, vis_type])
        
        tile = self.cfg.MODEL.MOSAIC_TILE[0] 
        cluster = str(self.cfg.MODEL.MOSAIC_CLUSTER)
                    
        raster_stats_file = os.path.join(self.output_mosaic_dir, f'raster_stats_{tile}_{cluster}_{self.max_distance_raster}_{self.cfg.DATASETS.BINARY_TRESHOLD}.csv')
        os.makedirs(os.path.dirname(raster_stats_file), exist_ok=True)
        
        with open(raster_stats_file, 'w') as raster_csv_file:                
            raster_writer = csv.writer(raster_csv_file)
            raster_writer.writerow(['filename', 'cluster', 'type', 'urban_type', 'max_dist', 'tn', 'fp', 'fn', 'tp', 'precision', 'recall', 'f1', 'vis_type'])
            
            if self.cfg.MODEL.USE_BRK: type_ = 'brk'
                        
            pred_path = os.path.join(self.output_mosaic_dir, f"{tile}_{cluster}_{type_}.tif")
            if not os.path.exists(pred_path):
                print(f'Pred raster {pred_path} not found, skipping evaluation.')
                return
            pred = rasterio.open(pred_path).read(1)

            for vis_type in ['', '_vis_', '_inv_']:
                gt_path = os.path.join(self.output_mosaic_dir, f"{tile}_{cluster}_{type_}_gt_{vis_type}.tif")
                try:
                    gt = rasterio.open(gt_path).read(1)
                except rasterio.errors.RasterioIOError:
                    print(f'GT raster {gt_path} not found, skipping.')
                    continue
                
                if self.cfg.DATASETS.SELECT_URBAN and self.cfg.DATASETS.SELECT_RURAL: 
                    u_type = ['_', 'urban', 'rural']
                else:
                    u_type = ['_']
                
                for u_t in u_type:
                    if u_t == 'urban':
                        urban_polygons = self.images_urban_rural[self.images_urban_rural['urban']]
                        urban_polygons = urban_polygons.dissolve()
                        pred_urban = pred.copy()
                        urban_mask = rasterio.features.geometry_mask(urban_polygons.geometry, transform=rasterio.open(pred_path).transform, invert=False, all_touched=True, out_shape=pred_urban.shape)
                        gt_urban = gt.copy()
                        
                        pred_urban = pred_urban.astype(float)
                        gt_urban = gt_urban.astype(float)

                        pred_urban[urban_mask] = np.nan
                        gt_urban[urban_mask] = np.nan
                        
                        # print(f'Computing urban metrics for {tile}, {cluster}, {type_}, {vis_type}, {u_t}')
                        # print(pred_urban.shape, gt_urban.shape)
                        # print(f'Number of NaN in pred_urban: {np.isnan(pred_urban).sum()}, number of non-NaN: {(~np.isnan(pred_urban)).sum()}')
                        # print(f'Number of NaN in gt_urban: {np.isnan(gt_urban).sum()}, number of non-NaN: {(~np.isnan(gt_urban)).sum()}')
                        
                        compute_raster_metrics(pred_urban, gt_urban, self.cfg, tile, cluster, type_, vis_type, u_t, raster_writer, self.max_distance_raster)
                    elif u_t == 'rural':                       
                        rural_polygons = self.images_urban_rural[~self.images_urban_rural['urban']]
                        rural_polygons = rural_polygons.dissolve()
                        pred_rural = pred.copy()
                        rural_mask = rasterio.features.geometry_mask(rural_polygons.geometry, transform=rasterio.open(pred_path).transform, invert=False, all_touched=True, out_shape=pred_rural.shape)
                        gt_rural = gt.copy()
                        
                        pred_rural = pred_rural.astype(float)
                        gt_rural = gt_rural.astype(float)
                        
                        pred_rural[rural_mask] = np.nan
                        gt_rural[rural_mask] = np.nan
                            
                        # print(f'Computing rural metrics for  {tile}, {cluster}, {type_}, {vis_type}, {u_t}')
                        # print(pred_rural.shape, gt_rural.shape)
                        # print(f'Number of NaN in pred_rural: {np.isnan(pred_rural).sum()}, number of non-NaN: {(~np.isnan(pred_rural)).sum()}')
                        # print(f'Number of NaN in gt_rural: {np.isnan(gt_rural).sum()}, number of non-NaN: {(~np.isnan(gt_rural)).sum()}')
                        
                        compute_raster_metrics(pred_rural, gt_rural, self.cfg, tile, cluster, type_, vis_type, u_t, raster_writer, self.max_distance_raster)
                    else:  
                        compute_raster_metrics(pred.copy(), gt.copy(), self.cfg, tile, cluster, type_, vis_type, u_t, raster_writer, self.max_distance_raster)
                            
        raster_csv_file.close()
    
             
class PredictVector():
    def __init__(self, cfg, output_dir):  
        self.cfg = cfg
        self.output_dir = output_dir
        self.output_mosaic_dir = os.path.normpath(os.path.join(self.output_dir, self.cfg.DATASETS.MOSAIC_DIR))
        self.output_dir_poly = os.path.join(self.output_mosaic_dir, f'{self.cfg.MODEL.TYPE_POLY_PRED}_{self.cfg.DATASETS.BINARY_TRESHOLD}_{self.cfg.POLY.TOLERANCE}')
        self.patch_resolution = cfg.DATASETS.PATCHES_DIR.split('_')[-1]
        dataset_name = os.path.basename(self.cfg.DATASETS.DATA_DIR)
        
        base = '' if not self.cfg.MODEL.TEST_OVERLAP_50 else "overlap_50/"
        if cfg.MODEL.USE_BRK:
            self.fname = f"{base}reference_{dataset_name}_ps_{self.patch_resolution}_brk.gpkg"

        self.ref_root = os.path.join(self.cfg.DATASETS.DATA_DIR, self.cfg.DATASETS.REF_DIR, '')

        if self.cfg.DATASETS.USE_URBAN_RURAL:
            urban_rural_file = self.ref_root + f'{base}urban_{self.patch_resolution}.gpkg'
            self.images_urban_rural = gp.read_file(urban_rural_file)
        else:
            self.images_urban_rural = None

        if self.cfg.DATASETS.USE_CLUSTERS:
            cluster_fname = self.cfg.DATASETS.CLUSTER_FNAME if self.cfg.DATASETS.CLUSTER_FNAME else f'{base}urban_{self.patch_resolution}_clusters.gpkg'
            cluster_file = self.ref_root + cluster_fname
            self.clusters = gp.read_file(cluster_file).set_crs(f'EPSG:{self.cfg.DATASETS.CRS}')
            if 'cluster_id' not in self.clusters.columns:
                self.clusters['cluster_id'] = '0'
        else:
            self.clusters = None
        
        if self.output_dir_poly:
            if os.path.isdir(self.output_dir_poly): pass
            else: os.makedirs(self.output_dir_poly, exist_ok=True)

    def write(self): 
        tile = self.cfg.MODEL.MOSAIC_TILE[0]
        cluster = str(self.cfg.MODEL.MOSAIC_CLUSTER)
        
        for type_ in ['brk']:
            if not getattr(self.cfg.MODEL, f"USE_{type_.upper()}"):
                continue
                            
            pred_path = os.path.join(self.output_mosaic_dir, f"{tile}_{cluster}_{type_}.tif")   
            try: mosaic = rasterio.open(pred_path)
            except: continue
            
            left = mosaic.bounds[0]
            bottom = mosaic.bounds[1]
            mosaic = mosaic.read(1)                    
            mosaic = np.expand_dims(np.expand_dims(mosaic, axis=0).astype(np.float32), axis=0)
                                
            cluster_height = mosaic.shape[2]
            
            vectorizer = VectorizerSimple(self.cfg, file_output_dir=self.output_dir_poly)
            polylines_batch = vectorizer(mosaic)

            out_polylines = []
            for polylines in polylines_batch:
                for polyline in polylines:
                    line_string = shapely.geometry.LineString(polyline[:, ::-1])
                    # line_string = line_string.simplify(self.cfg.POLY.TOLERANCE, preserve_topology=False)
                    out_polylines.append(line_string) 
            pred_polylines = gp.GeoDataFrame({'geometry':out_polylines}, geometry='geometry')  

            pixel_size = int(self.patch_resolution) / 100
            swapped_pred_polylines = []
            for line in pred_polylines.geometry:
                try:
                    swapped_coords = [(point[0], cluster_height - point[1]) for point in list(line.coords)] 
                    scaled_coords = [shapely.geometry.Point((point[0] * pixel_size) + left, (point[1] * pixel_size) + bottom) for point in swapped_coords] 
                    swapped_line = shapely.geometry.LineString(scaled_coords)
                    swapped_pred_polylines.append(swapped_line) 
                except Exception as e:
                    print(e)
                    continue
                
            rd_pred_polylines = gp.GeoDataFrame({'geometry':swapped_pred_polylines}, geometry='geometry', crs=f'EPSG:{self.cfg.DATASETS.CRS}')
            rd_pred_polylines['id'] = rd_pred_polylines.index + 1
            print(f'Writing predicted polylines for {tile} {cluster} {type_}')
            rd_pred_polylines.to_file(os.path.join(self.output_dir_poly, f'pred_polylines_{tile}_{cluster}_{type_}.gpkg'), driver="GPKG")
            
            print(f'Simplifying predicted polylines for {tile} {cluster} {type_} with tolerance {self.cfg.POLY.TOLERANCE}')
            simplified_pred_polylines = []
            for line in rd_pred_polylines.geometry:
                try:
                    simplified_line = line.simplify(self.cfg.POLY.TOLERANCE, preserve_topology=False)
                    simplified_pred_polylines.append(simplified_line)
                except Exception as e:
                    print(e)
                    simplified_pred_polylines.append(line)

            rd_pred_polylines['geometry'] = simplified_pred_polylines
            rd_pred_polylines.to_file(os.path.join(self.output_dir_poly, f'simplified_pred_polylines_{tile}_{cluster}_{type_}.gpkg'),
                driver="GPKG"
            )
            
    def calculate_vector_metrics(self, pred_polylines_, gt_polylines_, tile, cluster, u_type_, type_, vis_type, vector_writer, segment_length=0.4, max_distance=500):
        pred_feature_count = len(pred_polylines_)
        pred_feature_length = pred_polylines_.length.sum()    
        
        print("calculate Graph connectivity for predicted polylines")
        connectivity_graph = nx.Graph()
        for idx, geom in pred_polylines_.geometry.items():
            if isinstance(geom, LineString) and not geom.is_empty:
                coords = list(geom.coords)
                start = tuple(coords[0])
                end = tuple(coords[-1])
                connectivity_graph.add_edge(start, end, index=idx)

        pred_feature_connected_components = nx.number_connected_components(connectivity_graph)
        print(f'Number of connected components in predicted polylines: {pred_feature_connected_components}')
        
        pred_polylines_ = pred_polylines_.explode(index_parts=False)
        gt_polylines_ = gt_polylines_.explode(index_parts=False)

        rd_pred_buffer = pred_polylines_.copy()
        rd_gt_buffer = gt_polylines_.copy()

        rd_gt_buffer['geometry'] = rd_gt_buffer.buffer(0.4)
        # rd_gt_buffer.to_file(os.path.join(self.output_dir_poly, f'gt_buffer_{type_}_{vis_type}.gpkg'), driver="GPKG")
        
        rd_gt_buffer = rd_gt_buffer.dissolve()
        rd_gt_buffer['gt_area'] = rd_gt_buffer.geometry.area
        area_sum = rd_gt_buffer['gt_area'].sum()
        print(f"gt_area: {area_sum}")

        rd_pred_buffer['geometry'] = rd_pred_buffer.buffer(0.4)
        # rd_pred_buffer.to_file(os.path.join(self.output_dir_poly, f'pred_buffer_{type_}_{vis_type}.gpkg'), driver="GPKG")
        
        rd_pred_buffer = rd_pred_buffer.dissolve()
        rd_pred_buffer['pred_area'] = rd_pred_buffer.geometry.area
        print(f"pred_area: {rd_pred_buffer['pred_area'].sum()}")
        
        clipped_gt = rd_gt_buffer.clip(rd_pred_buffer)
        clipped_gt = clipped_gt.dissolve()
        clipped_gt['clip_gt_area'] = clipped_gt.geometry.area
        print(f"clip_gt_area: {clipped_gt['clip_gt_area'].sum()}")

        clipped_preds = rd_pred_buffer.clip(rd_gt_buffer)
        clipped_preds = clipped_preds.dissolve()
        clipped_preds['clip_pred_area'] = clipped_preds.geometry.area
        print(f"clip_pred_area: {clipped_preds['clip_pred_area'].sum()}")

        vector_recall = clipped_gt['clip_gt_area'].sum() / rd_gt_buffer['gt_area'].sum()
        vector_precision = clipped_preds['clip_pred_area'].sum() / rd_pred_buffer['pred_area'].sum()
        vector_f1 = 2 * (vector_precision * vector_recall) / (vector_precision + vector_recall)
        print(f"vector_precision: {vector_precision}")
        print(f"vector_recall: {vector_recall}")
        print(f"vector_f1: {vector_f1}")
        
        for idx, geom in gt_polylines_['geometry'].items():
            coords = list(geom.coords) if hasattr(geom, 'coords') else []
            if len(coords) < 2:
                print(f"GT geometry at index {idx} has less than 2 coords: {coords}")
                
        for idx, geom in pred_polylines_['geometry'].items():
            coords = list(geom.coords) if hasattr(geom, 'coords') else []
            if len(coords) < 2:
                print(f"Pred geometry at index {idx} has less than 2 coords: {coords}")
                
        gt_polylines_['geometry'] = gt_polylines_['geometry'].apply(
            lambda geom: geom.segmentize(segment_length) if geom.geom_type == 'LineString' and len(set(geom.coords)) > 1 else geom)
        pred_polylines_['geometry'] = pred_polylines_['geometry'].apply(
            lambda geom: geom.segmentize(segment_length) if geom.geom_type == 'LineString' and len(set(geom.coords)) > 1 else geom)

        # apls_score = np.nan
        # if vis_type == '':
        #     apls_score = run_apls(gt_polylines_, pred_polylines_, output_dir=self.output_dir_poly)
        #     print(f'APLS score for predicted polylines: {apls_score}')

        print("Processing vector metrics: discrepancy calculation started.")
        gt_area = rd_gt_buffer['gt_area'].sum()
        pred_area = rd_pred_buffer['pred_area'].sum()

        print("Extracting coordinates from GT and prediction polylines.")
        gt_coords = gt_polylines_.get_coordinates()
        gt_points = gp.GeoDataFrame(
            {'geometry': gp.points_from_xy(gt_coords['x'], gt_coords['y'])},
            crs=f'EPSG:{self.cfg.DATASETS.CRS}')
        gt_points["id"] = np.arange(1, len(gt_points) + 1)

        pred_coords = pred_polylines_.get_coordinates()
        pred_points = gp.GeoDataFrame(
            {'geometry': gp.points_from_xy(pred_coords['x'], pred_coords['y'])},
            crs=f'EPSG:{self.cfg.DATASETS.CRS}')
        pred_points["id"] = np.arange(1, len(pred_points) + 1)
        
        print(f"Number of ground truth points: {len(gt_points)}")
        print(f"Number of predicted points: {len(pred_points)}")

        print("Processing nearest neighbor matching from GT to prediction.")
        gt_nn = gt_points.sjoin_nearest(
            pred_points[['geometry', 'id']],
            how='left',
            distance_col="distances",
            max_distance=max_distance
        )
        gt_nn = gt_nn.merge(
            pred_points[['id', 'geometry']],
            left_on='id_right',
            right_on='id',
            suffixes=('_gt', '_pred')
        )
        
        print("Creating LineStrings from the original GT points to the matched prediction points.")                            
        gt_lines = [LineString([pt_gt, pt_pred]) for pt_gt, pt_pred in zip(gt_nn['geometry_gt'], gt_nn['geometry_pred'])]
        
        gt_nn = gt_nn.assign(geometry=gt_lines).set_geometry('geometry').drop(columns=['geometry_gt', 'geometry_pred'], errors='ignore')
        gt_nn['length'] = gt_nn.geometry.length
        # gt_nn.to_file(os.path.join(self.output_dir_poly, f'gt_nn_{type_}_{vis_type}_{u_type_}.gpkg'), driver="GPKG")                             
        
        print("Computing discrepancy area from gt lines.")        
        norm_gt_discrepancy = shapely.unary_union(shapely.buffer([gt_nn.geometry], 0.4)).area - min(gt_area, pred_area)

        print("Processing nearest neighbor matching from prediction to GT.")
        pred_nn = pred_points.sjoin_nearest(gt_points[['geometry', 'id']], how='left', distance_col="distances", max_distance=max_distance)
        pred_nn = pred_nn.merge(gt_points[['id', 'geometry']], left_on='id_right', right_on='id', suffixes=('_pred', '_gt'))
                
        print("Creating LineStrings from the original prediction points to the matched GT points.")                    
        pred_lines = [LineString([pt_pred, pt_gt]) for pt_pred, pt_gt in zip(pred_nn['geometry_pred'], pred_nn['geometry_gt'])]
        
        pred_nn = pred_nn.assign(geometry=pred_lines).set_geometry('geometry').drop(columns=['geometry_gt', 'geometry_pred'], errors='ignore')
        pred_nn['length'] = pred_nn.geometry.length
        # pred_nn.to_file(os.path.join(self.output_dir_poly, f'pred_nn_{type_}_{vis_type}_{u_type_}.gpkg'), driver="GPKG") 

        print("Computing discrepancy area from prediction lines.")
        norm_pred_discrepancy = shapely.unary_union(shapely.buffer([pred_nn.geometry], 0.4)).area - min(gt_area, pred_area)
        
        sum_normalized_discrepancy = norm_gt_discrepancy + norm_pred_discrepancy
        print(f"Normalized GT discrepancy: {norm_gt_discrepancy}")

        vector_writer.writerow([
            type_,
            vis_type,
            tile,
            cluster,
            u_type_,
            vector_precision,
            vector_recall,
            vector_f1,
            norm_gt_discrepancy,
            norm_pred_discrepancy,
            pred_feature_count,
            pred_feature_length,
            pred_feature_connected_components,
            sum_normalized_discrepancy,
            # apls_score
        ])
            
    def evaluate(self):      
        tile = self.cfg.MODEL.MOSAIC_TILE[0]
        cluster = str(self.cfg.MODEL.MOSAIC_CLUSTER)
        
        if self.cfg.MODEL.USE_BRK: type_ = 'brk'
        
        # Read predicted polylines from the specified file
        pred_poly_path = os.path.join(self.output_dir_poly, f'pred_polylines_{tile}_{cluster}_{type_}.gpkg')
        if not os.path.exists(pred_poly_path):
            print(f'Pred polylines {pred_poly_path} not found, skipping evaluation.')
            return
        rd_pred_polylines = gp.read_file(pred_poly_path)
        print(f'Total predicted polylines for tile {tile}, cluster {cluster}: {len(rd_pred_polylines)}')
        
        simplified_pred_polylines = gp.read_file(os.path.join(self.output_dir_poly, f'simplified_pred_polylines_{tile}_{cluster}_{type_}.gpkg'))
        print(f'Total simplified predicted polylines for tile {tile}, cluster {cluster}: {len(simplified_pred_polylines)}')
        
        if self.cfg.DATASETS.USE_CLUSTERS:
            tile_clusters = self.clusters[
                (self.clusters['image_id'].str.contains(tile, na=False)) &
                (self.clusters['cluster_id'] == cluster)
            ]
            tile_clusters_dissolved = gp.GeoDataFrame(tile_clusters.dissolve()).reset_index()
            clip_mask = box(*tile_clusters_dissolved.geometry.total_bounds)
        else:
            rd_pred_polylines_bounds = rd_pred_polylines.total_bounds
            clip_mask = box(*rd_pred_polylines_bounds)
        
        rd_gt_polylines = gp.read_file(os.path.join(self.ref_root, self.fname), engine="pyogrio", bbox=clip_mask)
        if 'filename_10k' not in rd_gt_polylines.columns and 'image_id' in rd_gt_polylines.columns:
            rd_gt_polylines['filename_10k'] = rd_gt_polylines['image_id'].str.extract(r'((?:tile-)?\d+-\d+)')
        if 'filename_10k' in rd_gt_polylines.columns:
            tile_numeric = tile.removeprefix('tile-')
            rd_gt_polylines = rd_gt_polylines[rd_gt_polylines['filename_10k'].str.contains(tile_numeric, na=False, regex=False)]
        
        rd_gt_polylines['visible'] = rd_gt_polylines['visible'].map({'True': True, 'False': False})
        print(f'Total ground truth polylines for tile {tile}, cluster {cluster}: {len(rd_gt_polylines)}')
        
        if self.cfg.DATASETS.USE_URBAN_RURAL:
            urban_polygons = self.images_urban_rural[self.images_urban_rural['urban']]
            urban_polygons = urban_polygons.dissolve()
            rural_polygons = self.images_urban_rural[~self.images_urban_rural['urban']]
            rural_polygons = rural_polygons.dissolve()
        else:
            urban_polygons = None
            rural_polygons = None  
                        
        print('Calculating vector metrics.')
        vector_stats_file = os.path.join(self.output_dir_poly, f'poly_stats_{tile}_{cluster}.csv')
        os.makedirs(os.path.dirname(vector_stats_file), exist_ok=True)
        with open(vector_stats_file, 'w') as vector_csv_file:
            vector_writer = csv.writer(vector_csv_file)
            vector_writer.writerow([
                'type',
                'vis_type',
                'tile',
                'cluster',
                'urban_type',
                'vector_precision',
                'vector_recall',
                'vector_f1',
                'norm_gt_discrepancy',
                'norm_pred_discrepancy',
                'pred_feature_count',
                'pred_feature_length',
                'pred_feature_connected_components',
                'sum_normalized_discrepancy',
                # 'apls'
            ])
    
            
            print(f'Processing vector metrics for tile {tile}, cluster {cluster}, type: {type_}')
            
            vis_types = ['', '_vis_', '_inv_']

            for vis_type in vis_types:                                                                
                if vis_type == '':
                    gt_polylines_visibility = rd_gt_polylines
                elif vis_type == '_vis_':
                    gt_polylines_visibility = rd_gt_polylines[rd_gt_polylines['visible']]
                elif vis_type == '_inv_':
                    gt_polylines_visibility = rd_gt_polylines[~rd_gt_polylines['visible']]  
                    if gt_polylines_visibility.empty:
                        continue
                            
                gt_polylines_visibility = gt_polylines_visibility.explode(index_parts=False)
                gt_polylines_visibility['id'] = gt_polylines_visibility.index + 1
                
                if self.cfg.DATASETS.SELECT_URBAN and self.cfg.DATASETS.SELECT_RURAL: u_type = ['_', 'urban', 'rural']
                else: u_type = ['_']
                
                for u_t in u_type:
                    if u_t == 'urban':
                        pred_urban = rd_pred_polylines.copy()
                        gt_urban = gt_polylines_visibility.copy()
                        pred_urban = pred_urban.clip(urban_polygons)
                        gt_urban = gt_urban.clip(urban_polygons)
                        print(f'Calculating vector metrics for urban areas.')
                        print(f'Number of predicted polylines for urban areas: {len(pred_urban)}, ground truth polylines: {len(gt_urban)}')
                        self.calculate_vector_metrics(pred_urban, gt_urban, tile, cluster, u_t, type_, vis_type, vector_writer)    
                    elif u_t == 'rural':                   
                        pred_rural = rd_pred_polylines.copy()
                        gt_rural = gt_polylines_visibility.copy()
                        pred_rural = pred_rural.clip(rural_polygons)
                        gt_rural = gt_rural.clip(rural_polygons)
                        print(f'Calculating vector metrics for rural areas.')
                        print(f'Number of predicted polylines for rural areas: {len(pred_rural)}, ground truth polylines: {len(gt_rural)}')
                        self.calculate_vector_metrics(pred_rural, gt_rural, tile, cluster, u_t, type_, vis_type, vector_writer)
                    else:
                        pred = rd_pred_polylines.copy()
                        gt = gt_polylines_visibility.copy()
                        print(f'Calculating vector metrics for all areas.')
                        print(f'Number of predicted polylines: {len(pred)}, ground truth polylines: {len(gt)}')
                        self.calculate_vector_metrics(pred, gt, tile, cluster, u_t, type_, vis_type, vector_writer)
        vector_csv_file.close()