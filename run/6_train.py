"""Training entry point — forwards to run.py with RUN_TYPE=train.
Usage: python run/6_train.py --config config/unet/cadnet_brk_unet_train.yaml
"""
import runpy, sys
sys.argv[0] = 'run/run.py'
runpy.run_path('run/run.py', run_name='__main__')
