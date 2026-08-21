import rasterio
import os
import math
import multiprocessing as mp
import time

from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.transform import Affine

os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "YES"
os.environ["GDAL_CACHEMAX"] = "32" 


def _process_single_image(args):
    image_path, output_dir, image_size, use_overlap, downsample_factor = args
    image_name = os.path.basename(image_path)
        
    counts = {
        "total_windows": 0,
        "written": 0,
        "exists": 0,
        "read_fail": 0,
        "write_fail": 0,
        "skipped_small": 0,
    }
    
    start_time = time.time()
    print(
        f"Start {image_name}: downsample_factor={downsample_factor} "
        f"image_size={image_size} overlap={use_overlap}"
    )

    def _patch_grid(width, height):
        if width < image_size or height < image_size:
            counts["skipped_small"] += 1
            print(
                f"Skip {image_name}: {width}x{height} < {image_size}"
            )
            return

        if use_overlap:
            stride_x = image_size // 2
            stride_y = image_size // 2

            patches_x = math.ceil((width - image_size) / stride_x) + 1
            patches_y = math.ceil((height - image_size) / stride_y) + 1
        else:
            patches_x = math.ceil(width / image_size)
            patches_y = math.ceil(height / image_size)

            patch_step_x_mod = (image_size * patches_x) % width
            patch_step_y_mod = (image_size * patches_y) % height

            stride_x = (
                image_size - int(patch_step_x_mod / (patches_x - 1))
                if patches_x > 1 else image_size
            )
            stride_y = (
                image_size - int(patch_step_y_mod / (patches_y - 1))
                if patches_y > 1 else image_size
            )

        return patches_x, patches_y, stride_x, stride_y

    def _write_patch(img, window, profile):
        out_name = f"image_{window.col_off}_{window.row_off}_{image_name}"
        out_path = os.path.join(output_dir, out_name)

        if os.path.exists(out_path):
            counts["exists"] += 1
            return

        try:
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(img)
            counts["written"] += 1
        except Exception as e:
            counts["write_fail"] += 1
            print(f"Write failed {out_path}: {e}")

    def _process_full_res(img):
        grid = _patch_grid(img.width, img.height)
        if grid is None:
            return
        patches_x, patches_y, stride_x, stride_y = grid
        last_x = patches_x - 1
        last_y = patches_y - 1

        for i in range(patches_x):
            for n in range(patches_y):
                counts["total_windows"] += 1
                col = (
                    img.width - image_size
                    if i == last_x
                    else i * stride_x
                )
                row = (
                    img.height - image_size
                    if n == last_y
                    else n * stride_y
                )

                w = Window(col, row, image_size, image_size)

                try:
                    data = img.read(window=w)
                except Exception as e:
                    counts["read_fail"] += 1
                    print(f"Read failed {image_name} @ {w}: {e}")
                    continue

                transform = img.window_transform(w)
                profile = img.profile.copy()
                profile.update({
                    "height": image_size,
                    "width": image_size,
                    "transform": transform
                })

                _write_patch(data, w, profile)

    def _process_downsampled(img):
        ds_width = int(img.width * downsample_factor)
        ds_height = int(img.height * downsample_factor)

        grid = _patch_grid(ds_width, ds_height)
        if grid is None:
            return
        patches_x, patches_y, stride_x, stride_y = grid
        last_x = patches_x - 1
        last_y = patches_y - 1

        scale = 1 / downsample_factor

        for i in range(patches_x):
            for n in range(patches_y):
                counts["total_windows"] += 1
                col = (
                    ds_width - image_size
                    if i == last_x
                    else i * stride_x
                )
                row = (
                    ds_height - image_size
                    if n == last_y
                    else n * stride_y
                )

                src_col = int(round(col * scale))
                src_row = int(round(row * scale))
                src_w = int(round(image_size * scale))
                src_h = int(round(image_size * scale))

                if src_col + src_w > img.width:
                    src_col = max(0, img.width - src_w)
                if src_row + src_h > img.height:
                    src_row = max(0, img.height - src_h)

                w = Window(src_col, src_row, src_w, src_h)

                try:
                    data = img.read(
                        window=w,
                        out_shape=(img.count, image_size, image_size),
                        resampling=Resampling.nearest,
                    )
                except Exception as e:
                    counts["read_fail"] += 1
                    print(f"Read failed {image_name} @ {w}: {e}")
                    continue

                transform = img.window_transform(w)
                transform = transform * Affine.scale(
                    w.width / image_size,
                    w.height / image_size,
                )

                profile = img.profile.copy()
                profile.update({
                    "height": image_size,
                    "width": image_size,
                    "transform": transform
                })

                _write_patch(data, w, profile)

    with rasterio.open(image_path) as img:
        if downsample_factor != 1:
            _process_downsampled(img)
        else:
            _process_full_res(img)

    elapsed = time.time() - start_time
    print(
        f"Done {image_name}: windows={counts['total_windows']} "
        f"written={counts['written']} exists={counts['exists']} "
        f"read_fail={counts['read_fail']} write_fail={counts['write_fail']} "
        f"skipped_small={counts['skipped_small']} "
        f"elapsed_s={elapsed:.2f}"
    )
                        

class CreatePatches:
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_dir = os.path.join(cfg.DATA_DIR, "train", cfg.IMAGES_DIR)
        self.validate_dir = os.path.join(cfg.DATA_DIR, "validate", cfg.IMAGES_DIR)
        self.test_dir = os.path.join(cfg.DATA_DIR, "test", cfg.IMAGES_DIR)
        
        self.root_dirs = [
            self.train_dir,
            self.validate_dir,
            self.test_dir
        ]

        self.image_size = cfg.IMG_SIZE
        self.test_overlap = cfg.TEST_OVERLAP_50

    def create_patch(self):
        cpu_count = mp.cpu_count()
        max_workers = min(16, max(1, cpu_count))

        for root_folder in self.root_dirs:
            split_name = os.path.basename(os.path.dirname(root_folder))
            split_start = time.time()
            print(
                f"Split start: {split_name} root={root_folder} "
                f"test_overlap={self.test_overlap}"
            )
            
            use_overlap = (split_name == "test") and self.test_overlap

            split_path = os.path.split(root_folder)[0]
            patches_dir = f'{self.cfg.PATCHES_DIR}_overlap_50' if use_overlap else self.cfg.PATCHES_DIR
            output_dir = os.path.join(
                split_path,
                patches_dir,
                self.cfg.IMAGES_DIR
            )
            os.makedirs(output_dir, exist_ok=True)

            image_paths = [
                os.path.join(root_folder, f)
                for f in os.listdir(root_folder)
            ]

            if not image_paths:
                print(
                    f"Split {split_name}: no images found in {root_folder}; "
                    f"skipping."
                )
                continue

            args = [
                (
                    img_path,
                    output_dir,
                    self.image_size,
                    use_overlap,
                    self.cfg.DOWNSAMPLE_FACTOR,
                )
                for img_path in image_paths
            ]
            
            print(
                f"Split {split_name}: images={len(image_paths)} "
                f"output={output_dir}"
            )
                
            with mp.Pool(processes=max_workers, maxtasksperchild=4) as p:
                p.map(_process_single_image, args)

            split_elapsed = time.time() - split_start
            print(f"Split done: {split_name} elapsed_s={split_elapsed:.2f}")
            
        