from pathlib import Path

BASE = Path(__file__).parents[1]
ROOT = BASE

HOME = Path.home()
MANO_DIR = HOME / "_DATA/data/"
MANO_CFG = {
    # "data_dir": MANO_DIR,
    "model_path": MANO_DIR / "mano",
    "gender": "neutral",
    "num_hand_joints": 15,
    "mean_params": MANO_DIR / "mano_mean_params.npz",
    "create_body_pose": False,
}
