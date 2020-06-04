import os
import json

import sys
sys.path.append("../")
from utils import dict_product, iwt

with open("../MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["Humanoid-v2", "Walker2d-v2", "Hopper-v2", "Swimmer-v2", "Reacher-v2", "Pusher-v2", "HalfCheetah-v2", "Ant-v2"],
    "mode": ["trpo"],
    "out_dir": ["trust_region_study/agents"],
    "norm_rewards": ["returns"],
    "initialization": ["orthogonal"],
    "anneal_lr": [True],
    "value_clipping": [False],
    "ppo_lr_adam": [1e-4],
    "val_lr": [5e-5, 1e-4, 3e-4],
    "advanced_logging": [True],
    "cpu": [True],
    "use_cons": ['all', 'kl', 'rew', 'none'],
    "seed": [1,2,3,4,5],
    "use_conj": [True, False]
}

all_configs = [{**BASE_CONFIG, **p} for p in dict_product(PARAMS)]
if os.path.isdir("agent_configs/") or os.path.isdir("agents/"):
    raise ValueError("Please delete the 'agent_configs/' and 'agents/' directories")
os.makedirs("agent_configs/")
os.makedirs("agents/")

for i, config in enumerate(all_configs):
    with open(f"agent_configs/{i}.json", "w") as f:
        json.dump(config, f)
