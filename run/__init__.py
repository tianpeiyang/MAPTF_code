from .run_multi_ptf_ppo_sro import run as multi_ppo_sr_run
from .run_maddpg_sr import run as run_maddpg_sr
from .run_multi_ptf_shppo_sro import run as shppo_sr_run

REGISTRY = {}

REGISTRY['multi_ppo'] = multi_ppo_sr_run
REGISTRY['multi_ppo_sro'] = multi_ppo_sr_run
REGISTRY['maddpg'] = run_maddpg_sr
REGISTRY['maddpg_sr'] = run_maddpg_sr
REGISTRY['shppo'] = shppo_sr_run
REGISTRY['shppo_sro'] = shppo_sr_run

