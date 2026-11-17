"""Fresh/resume controlled training. Invocation requires an existing validated seal."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from drone_detection.yolo26_workflow import main
if __name__=='__main__':main()
