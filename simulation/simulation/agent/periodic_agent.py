import random

import numpy

from .agent_base import AgentBase


class PeriodicAgent(AgentBase):
    def reset(self):
        self.target_idx = random.randint(0, len(self.config.targets) - 1)
        super().reset()

    def _gen_target(self):
        x, y = self.config.targets[self.target_idx]
        return numpy.array([x, y])

    def random_position(self):
        self.p = self._gen_target()

    def reset_target(self):
        self.target_idx = (self.target_idx + 1) % len(self.config.targets)
        self.target = self._gen_target()
