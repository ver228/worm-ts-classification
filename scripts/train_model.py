import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from worm_ts_classification.trainer import Trainer

def main(**argkws):
    tt = Trainer(**argkws)
    tt.train()
    
        
if __name__ == '__main__':
    import fire
    fire.Fire(main)
    