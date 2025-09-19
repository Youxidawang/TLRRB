"""
Training script for TLRRB
"""
from run import TLRRB_run
if __name__ == '__main__':
    for i in range(1):
        TLRRB_run(model_name='TLRRB', dataset_name='mosi', is_tune=False, seeds=[], model_save_dir="./pt",
                 res_save_dir="./result", log_dir="./log", mode='train', is_training=True)

