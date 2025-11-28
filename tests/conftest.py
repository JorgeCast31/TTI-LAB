import sys
from pathlib import Path

# Add project root to sys.path so that core, simulations, etc. can be imported
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
