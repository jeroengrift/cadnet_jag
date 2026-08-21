"""Batch prediction entry point — forwards to predict_batch.py.
Usage: python run/7_predict_batch.py --input_dir <dir> [options]
"""
import runpy, sys
sys.argv[0] = 'run/predict_batch.py'
runpy.run_path('run/predict_batch.py', run_name='__main__')
