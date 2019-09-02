import numpy as np
from quantum_agent import QuantumAgent

from lib.envs.simple_rooms import SimpleRoomsEnv
from lib.envs.windy_gridworld import WindyGridworldEnv
from lib.envs.cliff_walking import CliffWalkingEnv
from lib.simulation import Experiment

if __name__ == '__main__':
    interactive = False
    env = SimpleRoomsEnv()
    agent = QuantumAgent(range(env.action_space.n))
    experiment = Experiment(env, agent)
    experiment.run_qlearning(50, interactive)