import numpy

from .field_object import FieldObject


class StayAgent(FieldObject):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.angle = 0

    def reset(self):
        super().reset()

    def step(self):

        return numpy.array([0.0, 0.0])
