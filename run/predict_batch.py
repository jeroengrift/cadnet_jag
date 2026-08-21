import os
import sys
import subprocess
import argparse
import yaml
import geopandas as gp
from concurrent.futures import ThreadPoolExecutor, as_completed


def str2bool(value):
    """Parse a boolean value from a CLI argument."""
    if isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    if val in ('yes', 'true', 't', 'y', '1'):
        return True
    if val in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")

def run_command(cmd, env=None):
    print("Starting:", " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    return result.returncode

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run predict all tiles')
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--cluster_file', type=str, default=r'data/nl/reference/overlap_50/urban_25_clusters.gpkg')
    parser.add_argument('--max_parallel_jobs', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_dist', type=float, default=8.0)
    parser.add_argument('--tolerance', type=float, default=2.0)
    parser.add_argument('--buffer', type=float, default=0.0)
    parser.add_argument('--mosaic_dir', type=str, default='mosaic_preds_no_buffer')
    parser.add_argument('--tiles', type=str, nargs='*', default=None,
                        help='Override tile list; if omitted the hardcoded list is used.')

    # Logging/output flags
    parser.add_argument('--write_mosaic_raster', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--write_raster_stats', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--write_mosaic_vector', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--write_vector_stats', type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()

    input_dir = args.input_dir
    cluster_file = gp.read_file(args.cluster_file).set_crs('EPSG:28992')
    cluster_file = cluster_file[cluster_file['split'] == 'test']
    
    generated_dir = os.path.join(input_dir, 'tile_configs')
        
    os.makedirs(generated_dir, exist_ok=True)
    
    max_parallel_jobs = args.max_parallel_jobs
    commands = []
    
    list_dir = os.listdir(input_dir)
    
    for filename in list_dir:
        if not filename.endswith('.yaml'):
                continue

        cfg_path = os.path.join(input_dir, filename)
        tiles = args.tiles if args.tiles is not None else [
            '20000-370000',
            '90000-470000',
            '110000-460000',
            '120000-430000',
            '140000-420000',
            '170000-370000',
            '170000-460000',
            '170000-550000',
            '180000-520000',
            '190000-310000',
            '190000-400000',
            '200000-540000',
            '240000-590000',
        ]
        
        for tile in tiles:
            cluster_tile = cluster_file[cluster_file['image_id'].str.contains(tile)]
            cluster_ids = cluster_tile['cluster_id'].dropna().unique().tolist()
            
            for cluster_id in cluster_ids:
                print(f'Preparing tile {tile} cluster {cluster_id} from config {filename}')
                
                with open(cfg_path, 'r') as f:
                    cfg = yaml.safe_load(f)
                    cfg['MODEL']['MOSAIC_TILE'] = [tile]
                    cfg['MODEL']['MOSAIC_CLUSTER'] = cluster_id
                    cfg['MODEL']['DEVICE'] = 'cuda:0'  # CUDA_VISIBLE_DEVICES limits to the right physical GPU
                    cfg['DATALOADER']['NUM_WORKERS'] = args.num_workers
                    cfg['MODEL']['WRITE_MOSAIC_RASTER'] = args.write_mosaic_raster
                    cfg['MODEL']['WRITE_RASTER_STATS'] = args.write_raster_stats
                    cfg['MODEL']['WRITE_MOSAIC_VECTOR'] = args.write_mosaic_vector
                    cfg['MODEL']['WRITE_VECTOR_STATS'] = args.write_vector_stats
                    cfg['MODEL']['MAX_DIST'] = args.max_dist
                    cfg['POLY']['TOLERANCE'] = args.tolerance
                    cfg['DATASETS']['MOSAIC_DIR'] = args.mosaic_dir
                    cfg['DATASETS']['BUFFER'] = args.buffer

                    base, ext = os.path.splitext(filename)
                    tmp_cfg_path = os.path.join(generated_dir, f"{base}_{tile}_{int(cluster_id)}_{int(args.max_dist)}_{int(args.tolerance)}{ext}")
                    with open(tmp_cfg_path, 'w') as tmpf:
                        yaml.safe_dump(cfg, tmpf)

                    cmd = [
                        "python",
                        "run/run.py",
                        f"--config={tmp_cfg_path}"
                    ]
                    commands.append(cmd)

    # Restrict each subprocess to only the requested GPU so no CUDA context
    # is accidentally created on GPU 0.
    gpu_id = args.device.split(':')[-1] if ':' in args.device else '0'
    subprocess_env = os.environ.copy()
    subprocess_env['CUDA_VISIBLE_DEVICES'] = gpu_id

    with ThreadPoolExecutor(max_workers=max_parallel_jobs) as executor:
        futures = [executor.submit(run_command, cmd, subprocess_env) for cmd in commands]
        for future in as_completed(futures):
            retcode = future.result()
            if retcode != 0:
                print(f"Command failed with return code {retcode}")

    print("All commands completed.")