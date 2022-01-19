import numpy

from .agent_base import AgentBase


class RandomAgent(AgentBase):

    def reset_target(self):
        boundary = self.world.get_boundary()
        self.target = numpy.array([
            numpy.random.uniform(boundary[0][0], boundary[0][1]),
            numpy.random.uniform(boundary[1][0], boundary[1][1]),
        ])
