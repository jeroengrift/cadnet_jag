import os
import csv
import numpy as np
from sklearn.metrics import confusion_matrix
import rasterio
import pandas as pd
from multiprocessing import Pool, cpu_count
import geopandas as gpd

output_dir = 'output/jag_2026'

def read_raster():
    output_csv_path = os.path.join(output_dir, "predict_raster.csv")
    with open(output_csv_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["model", "tile", "cluster", "type", "urban_type", "tn", "fp", "fn", "tp", "precision", "recall", "f1", "vis_type", "bin_threshold", "max_distance"])

        for dirpath, dirnames, filenames in os.walk(output_dir):
            if 'mosaic_preds_no_buffer' in dirpath:
                for filename in filenames:
                    if filename.endswith('.csv'):
                        if filename.startswith('raster'):
                            model = dirpath.split('/')[-2].split('-')[2]
                            bin_threshold = filename.split('_')[-1].replace('.csv', '')
                            max_distance = filename.split('_')[-2]
                            cluster = filename.split('_')[-3]
                            try:
                                with open(os.path.join(dirpath, filename), "r") as preds:
                                    reader = csv.reader(preds)
                                    data = list(reader)[1:]
                                    for line in data:
                                        tile = line[0]    
                                        cluster = line[1]                                    
                                        type_ = line[2]
                                        urban_type = line[3]
                                        tn = line[5]
                                        fp = line[6]
                                        fn = line[7]
                                        tp = line[8]
                                        precision = line[9]
                                        recall = line[10]
                                        f1 = line[11]
                                        vis = line[12]                                                 
                                        csvwriter.writerow([model, tile, cluster, type_, urban_type, tn, fp, fn, tp, precision, recall, f1, vis, bin_threshold, max_distance])

                            except Exception as e: 
                                continue

def read_vector(): 
    output_csv_path = os.path.join(output_dir, "predict_vector.csv")
    with open(output_csv_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow([
            "model",
            "tile",
            "cluster",
            "type",
            "vis_type",
            "urban_type",
            "vector_precision",
            "vector_recall",
            "vector_f1",
            "norm_gt_discrepancy",
            "norm_pred_discrepancy",
            "norm_discrepancy_sum",
            "pred_feature_count",
            "pred_feature_length",
            "pred_feature_connected_components",
            "bin_threshold",
            "simplify_tolerance"
        ])
        
        for dirpath, dirnames, filenames in os.walk(output_dir):
            for filename in filenames:
                if filename.endswith('.csv'):
                    if filename.startswith('poly'):
                        model = dirpath.split('/')[-3].split('-')[2]  
                        tile = filename.split('_')[2]
                        cluster = filename.split('_')[3].split('.')[0]
                        bin_threshold = dirpath.split('/')[-1].split('_')[-2]
                        simplify_tolerance = dirpath.split('/')[-1].split('_')[-1]
                        try:
                            with open(os.path.join(dirpath, filename), "r") as preds:
                                reader = csv.reader(preds)
                                data = list(reader)[1:]    
                                for line in data:                           
                                    type_ = line[0]
                                    vis = line[1]
                                    urban_type = line[4]
                                    vector_precision = line[5]
                                    vector_recall = line[6]
                                    vector_f1 = line[7]
                                    norm_gt_discrepancy = line[8]
                                    norm_pred_discrepancy = line[9]
                                    pred_feature_count = line[10]
                                    pred_feature_length = line[11]
                                    pred_feature_connected_components = line[12]
                                    norm_discrepancy_sum = line[13]

                                    csvwriter.writerow([
                                        model,
                                        tile,
                                        cluster,
                                        type_,
                                        vis,
                                        urban_type,
                                        vector_precision,
                                        vector_recall,
                                        vector_f1,
                                        norm_gt_discrepancy,
                                        norm_pred_discrepancy,
                                        norm_discrepancy_sum,
                                        pred_feature_count,
                                        pred_feature_length,
                                        pred_feature_connected_components,
                                        bin_threshold,
                                        simplify_tolerance
                                    ])
                        except:
                            print('Error reading file')
                            continue    

def evaluate_raster():
    predict_raster_df = pd.read_csv(os.path.join(output_dir, "predict_raster.csv"))
    predict_raster_df['vis_type'] = predict_raster_df['vis_type'].fillna('all')

    stats = predict_raster_df.groupby(['model', 'urban_type', 'bin_threshold', 'vis_type', 'max_distance']).agg({
        'tn': ['sum'],
        'fp': ['sum'],
        'fn': ['sum'],
        'tp': ['sum']
    }).reset_index()
        
    stats['precision'] = stats['tp'] / (stats['tp'] + stats['fp'])
    stats['recall'] = stats['tp'] / (stats['tp'] + stats['fn'])
    stats['f1'] = 2 * stats['precision'] * stats['recall'] / (stats['precision'] + stats['recall'])
            
    stats.columns = [
        f"{c[0]}_{c[1]}" if c[1] else c[0]
        for c in stats.columns.to_flat_index()
    ]   
    stats.to_csv(os.path.join(output_dir, "evaluate_raster.csv"), index=False)
    
def evaluate_vector():
    # only use cluster where all files are present (otherwise not comparable)
    predict_vector_df = pd.read_csv(os.path.join(output_dir, "predict_vector.csv"))    
    predict_vector_df['vis_type'] = predict_vector_df['vis_type'].fillna('all')
    predict_vector_df = predict_vector_df.dropna(subset=['vector_f1'])

    # clusters_df = predict_vector_df.groupby(['tile', 'cluster', 'urban_type', 'vis_type']).size().reset_index(name='model_count')
    # valid_clusters = clusters_df[clusters_df['model_count'] == 6][['tile', 'cluster', 'urban_type', 'vis_type']]
    # predict_vector_df = pd.merge(predict_vector_df, valid_clusters, on=['tile', 'cluster', 'urban_type', 'vis_type'], how='inner')
    
    stats = predict_vector_df.groupby(['model', 'urban_type', 'bin_threshold', 'vis_type']).agg({
        'vector_precision': ['mean'],
        'vector_recall':    ['mean'],
        'vector_f1':        ['mean'],
        'norm_gt_discrepancy':   ['sum'],
        'norm_pred_discrepancy': ['sum'],
        'norm_discrepancy_sum':   ['sum'],
        'pred_feature_count':    ['sum'],
        'pred_feature_length':   ['sum'],
        'pred_feature_connected_components': ['sum']
    }).reset_index()
    stats.columns = [
        f"{c[0]}_{c[1]}" if c[1] else c[0]
        for c in stats.columns.to_flat_index()
    ]

    ref_cc = pd.read_csv(os.path.join(os.path.dirname(__file__), 'ref_cc.csv'))
    stats = stats.merge(ref_cc, on=['urban_type', 'vis_type'], how='left')
    stats['cc_ratio'] = stats['pred_feature_connected_components_sum'] / stats['ref_feature_connected_components_sum']

    stats.to_csv(os.path.join(output_dir, "evaluate_vector.csv"), index=False)

def collect_vector():
    def collect_and_merge_gpkgs(suffix, name_part):
        gpkg_files = {}
        for dirpath, dirnames, filenames in os.walk(output_dir):
            for fname in filenames:
                if fname.startswith(suffix) and fname.endswith('.gpkg'):
                    parts = dirpath.split(os.sep)
                    model = '-'.join(parts[7].split('-')[:3])     
                    gpkg_files.setdefault(f'{model}', []).append(os.path.join(dirpath, fname))

        for model, paths in gpkg_files.items():
            gdfs = [gpd.read_file(p) for p in paths]
            if not gdfs:
                continue
            merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
            out_path = os.path.join(output_dir, f"vector_concat_{model}{name_part}.gpkg")
            merged.to_file(out_path, driver="GPKG")

    collect_and_merge_gpkgs('pred_polylines', '')
    collect_and_merge_gpkgs('simplified_pred_polylines', '_simplified')

if __name__ == '__main__':
    read_raster()
    read_vector()
    evaluate_raster()
    evaluate_vector()
    collect_vector()