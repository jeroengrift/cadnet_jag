import os
import shutil
import torch
import importlib
import argparse
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.default import cfg
from dataset.custom_dataset import collate_fn, CustomDatasetCad
from dataset import transforms
from pipeline.split import SplitDataset
from pipeline.patches import CreatePatches
from pipeline.train import Train
from pipeline.predict import PredictRaster, PredictVector

os.environ["CUDA_HOME"] = "/usr/local/cuda-12.6/lib64"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

# ----------------------------------------------------------------------------------------------
# step config
parser = argparse.ArgumentParser(description='Config file')
parser.add_argument("--config", type=str, default='')
args = parser.parse_args()

config_file = args.config
cfg.merge_from_file(config_file)
cfg.freeze()  

model_module = importlib.import_module(cfg.MODEL.PACKAGE)
MyModel = getattr(model_module, cfg.MODEL.CLASS_NAME)
device = torch.device(cfg.MODEL.DEVICE)

dataset_name = cfg.DATASETS.DATA_DIR.split('/')[-1]
output_dir = os.path.abspath(os.path.join(
    cfg.OUTPUT_DIR,
    cfg.DATASETS.REFERENCE_TYPE,
    (
        f"model-{cfg.MODEL.CLASS_NAME}"
        f"_bb-{cfg.MODEL.BACKBONE}"
    )
))

# ----------------------------------------------------------------------------------------------
# step split dataset into train, validate and test
if cfg.DATASETS.SPLIT:
    split_dataset_obj = SplitDataset(cfg=cfg)
    split_dataset_obj.split_dataset_file()

# ----------------------------------------------------------------------------------------------
# step create patches of 512x512
if cfg.DATASETS.CREATE_PATCHES:
    create_patches_obj = CreatePatches(cfg=cfg)
    create_patches_obj.create_patch()

# ----------------------------------------------------------------------------------------------
# step train the model
if cfg.MODEL.RUN_TYPE == 'train':
    dataset = 'train'
    train_dargs = dict()
    train_dargs['img_root'] = os.path.join(cfg.DATASETS.DATA_DIR, dataset, cfg.DATASETS.PATCHES_DIR, cfg.DATASETS.IMAGES_DIR)
    train_dargs['ref_root'] = os.path.join(cfg.DATASETS.DATA_DIR, cfg.DATASETS.REF_DIR)
    if cfg.MODEL.TEST_OVERLAP_50:
        train_dargs['ref_root'] = os.path.join(train_dargs['ref_root'], 'overlap_50')
    train_dargs['cfg'] = cfg
    train_dargs['run_type'] = cfg.MODEL.RUN_TYPE
    train_dargs['transform'] = transforms.ToTensor()
        
    train_dataset = CustomDatasetCad(**train_dargs)
    
    norm_train_data = torch.utils.data.DataLoader(train_dataset,
                                    batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                    collate_fn=collate_fn,
                                    shuffle=True,
                                    num_workers=cfg.DATALOADER.NUM_WORKERS,
                                    drop_last=True,
                                    worker_init_fn = np.random.seed(42))
    
    train_data = torch.utils.data.DataLoader(train_dataset,
                                batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                collate_fn=collate_fn,
                                shuffle=True,
                                num_workers=cfg.DATALOADER.NUM_WORKERS,
                                drop_last=True,
                                worker_init_fn = np.random.seed(42))

    dataset = 'validate'
    val_dargs = dict()
    val_dargs['img_root'] = os.path.join(cfg.DATASETS.DATA_DIR, dataset, cfg.DATASETS.PATCHES_DIR, cfg.DATASETS.IMAGES_DIR)
    val_dargs['ref_root'] = os.path.join(cfg.DATASETS.DATA_DIR, cfg.DATASETS.REF_DIR)
    if cfg.MODEL.TEST_OVERLAP_50:
        val_dargs['ref_root'] = os.path.join(val_dargs['ref_root'], 'overlap_50')
    val_dargs['cfg'] = cfg
    val_dargs['run_type'] = cfg.MODEL.RUN_TYPE
    val_dargs['transform'] = transforms.ToTensor()

    val_data = CustomDatasetCad(**val_dargs)
    val_data = torch.utils.data.DataLoader(val_data,
                                    batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                    collate_fn=collate_fn,
                                    shuffle=False,
                                    num_workers=cfg.DATALOADER.NUM_WORKERS,
                                    drop_last=True)
        
    if output_dir:
        if os.path.isdir(output_dir) and not cfg.MODEL.USE_PRETRAINED:
            raise ValueError(f"Output directory {output_dir} already exists.")
        elif os.path.isdir(output_dir) and cfg.MODEL.USE_PRETRAINED:
            pass
        elif not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    model = MyModel(cfg=cfg, arch=cfg.MODEL.CLASS_NAME, run_type=cfg.MODEL.RUN_TYPE)
    model = model.to(device)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Total parameters: {params}")

    if cfg.MODEL.USE_PRETRAINED:
        checkpoint = torch.load(os.path.join(output_dir, cfg.MODEL.TRAINED_MODEL))
        model.load_state_dict(checkpoint['model'])
    
    train = Train(cfg, output_dir, model, norm_train_data, train_data, val_data)
    train.train_model()

# ----------------------------------------------------------------------------------------------
# step create predictions
if cfg.MODEL.RUN_TYPE == 'predict':  
    img_root = os.path.join(cfg.DATASETS.DATA_DIR, 'test', cfg.DATASETS.PATCHES_DIR, cfg.DATASETS.IMAGES_DIR)
    ref_root = os.path.join(cfg.DATASETS.DATA_DIR, cfg.DATASETS.REF_DIR)
    
    if cfg.MODEL.TEST_OVERLAP_50:
        img_root = img_root.replace(cfg.DATASETS.PATCHES_DIR, f'{cfg.DATASETS.PATCHES_DIR}_overlap_50')
        ref_root_overlap = os.path.join(ref_root, 'overlap_50')
        # Some datasets (e.g. visible references) do not have a dedicated overlap_50
        # reference folder; in that case, use the base reference directory.
        ref_root = ref_root_overlap if os.path.isdir(ref_root_overlap) else ref_root
        
    if cfg.MODEL.WRITE_MOSAIC_RASTER:
        test_dargs = dict()
        test_dargs['img_root'] = img_root
        test_dargs['ref_root'] = ref_root
        test_dargs['cfg'] = cfg
        test_dargs['run_type'] = cfg.MODEL.RUN_TYPE
        test_dargs['transform'] = transforms.ToTensor()

        test_data = CustomDatasetCad(**test_dargs)
        test_data = torch.utils.data.DataLoader(test_data,
                                        batch_size=1,
                                        collate_fn=collate_fn,
                                        shuffle=False,
                                        num_workers=cfg.DATALOADER.NUM_WORKERS,
                                        drop_last=True)
        
        model = MyModel(cfg=cfg, arch=cfg.MODEL.CLASS_NAME, run_type=cfg.MODEL.RUN_TYPE)
        # Load checkpoint on CPU first to avoid GPU memory spikes during deserialization.
        checkpoint = torch.load(os.path.join(output_dir, cfg.MODEL.TRAINED_MODEL), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.eval()  
        model = model.to(device)  

        polygonize = PredictRaster(cfg, output_dir, model, test_data)
        polygonize.write()

    if cfg.MODEL.WRITE_RASTER_STATS:
        polygonize = PredictRaster(cfg, output_dir)
        polygonize.evaluate()
        
    if cfg.MODEL.WRITE_MOSAIC_VECTOR:
        polygonize = PredictVector(cfg, output_dir)
        polygonize.write()
        
    if  cfg.MODEL.WRITE_VECTOR_STATS:
        polygonize = PredictVector(cfg, output_dir)
        polygonize.evaluate()   
