from warehouse_env import WarehouseEnv, EnvConfig

from typing import List, Tuple, Dict, Any
from warehouse_env import WarehouseEnv, EnvConfig

# A single step result from env.step()
# (observation, reward, done, info)
Step = Tuple[Dict[str, Any], float, bool, Dict[str, Any]]

def rollout(seed: int, actions: list[int]) -> List[Step]:
    """
    Execute a fixed action sequence and record the full trajectory.

    Used to verify deterministic behavior. If the environment is truly
    deterministic, identical seeds and actions must produce identical
    trajectories.
    """
    env = WarehouseEnv(EnvConfig(width=5, height=5, num_packages=1, max_steps=50, battery_capacity=50))

    traj: List[Step] = []

    obs0 = env.reset(seed=seed)
    traj.append((obs0, 0.0, False, {"event": "reset"}))

    for a in actions:
        obs, r, done, info = env.step(a)
        traj.append((obs, r, done, info))
        if done:
            break

    return traj

def test_determinism_same_seed_same_traj():
    """
    Same seed and same actions must produce identical trajectories.
    """
    actions = [3, 3, 1, 1, 4, 3, 1, 5, 0, 2, 2]
    t1 = rollout(123, actions)
    t2 = rollout(123, actions)
    assert t1 == t2

def test_different_seed_usually_changes_reset():
    """
    Different seeds should normally produce different initial states.
    """
    env1 = WarehouseEnv(EnvConfig(width=5, height=5, num_packages=1))
    o1 = env1.reset(seed=1)
    env2 = WarehouseEnv(EnvConfig(width=5, height=5, num_packages=1))
    o2 = env2.reset(seed=2)
    assert o1 != o2
