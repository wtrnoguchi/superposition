import random

import numpy

from .field_object import FieldObject


class AgentBase(FieldObject):
    def __init__(self, config):
        self.config = config
        self.angle = 0

    def reset(self):
        super().reset()
        self.reset_target()
        self.is_sleep = False

    def is_goal(self):
        if numpy.linalg.norm(self.p - self.target) < self.config.goal_margin:
            return True
        else:
            return False

    def walk(self):
        v = self.target - self.p
        v /= numpy.linalg.norm(v)
        v *= self.config.v

        self.p = self.p + v

        return v[0], v[1]

    def step_no_sleep(self):

        vr, vl = self.walk()
        goal_flag = self.is_goal()
        if goal_flag:
            self.reset_target()

        return numpy.array([vr, vl]), goal_flag

    def step(self):

        if self.is_sleep:
            vr, vl = 0., 0.
            self.sleep_count -= 1
            if self.sleep_count == 0:
                self.is_sleep = False
        else:
            vr, vl = self.walk()
            if self.is_goal():
                self.reset_target()
            if random.random() < self.config.sleep_p:
                self.is_sleep = True
                self.sleep_count = self.config.sleep_base + \
                    random.randrange(- self.config.sleep_margin,
                                     self.config.sleep_margin + 1)

        return numpy.array([vr, vl])
