
from pathlib import Path

BASE = Path(__file__).parent.parent

MANO_DIR = BASE / "_DATA/data/"
MANO_CFG = {
    # "data_dir": MANO_DIR,
    "model_path": MANO_DIR / "mano",
    "gender": f"neutral",
    "num_hand_joints": 15,
    "mean_params": MANO_DIR / "mano_mean_params.npz",
    "create_body_pose": False,
}
