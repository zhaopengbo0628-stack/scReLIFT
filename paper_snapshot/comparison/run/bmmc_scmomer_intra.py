import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from bmmc_neural_intra import main
if __name__=='__main__':
    main('scmomer')
