import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neural_s_mode_intra import main

if __name__ == '__main__':
    main('eclare')
