import random

import numpy


class FieldObject(object):
    def __init__(self):
        pass

    def assign_world(self, world):
        self.world = world

    def reset(self):
        self.random_position()

    def random_position(self):
        bound = self.world.get_boundary()
        x = random.uniform(bound[0][0], bound[0][1])
        y = random.uniform(bound[1][0], bound[1][1])
        self.p = numpy.array([x, y])

    def get_position(self):
        return self.p

    def set_position(self, p):
        assert isinstance(p, numpy.ndarray)
        self.p = p

    def dist(self, o):
        return numpy.linalg.norm(o.p - self.p)
