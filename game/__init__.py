from .particle.make_env import make_env as Particle
from .pacman.make_env import make_env as PacmanEnv


REGISTRY = {}

REGISTRY['particle'] = Particle
REGISTRY['pacman'] = PacmanEnv


