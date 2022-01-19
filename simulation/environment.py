import copy

try:
    from . import creator
except Exception:
    import creator


class Environment(object):
    def __init__(self, config):

        self.world = creator.create_world(config.world)
        self.self_agent = creator.create_agent(config.agent.self)
        self.other_agent = creator.create_agent(config.agent.other)
        self.self_agent.assign_world(self.world)
        self.other_agent.assign_world(self.world)

    def init(self):
        self.world.init()

    def reset(self):
        self.self_agent.reset()
        self.other_agent.reset()

    def set_agent_pos(self, sp, op):
        self.self_agent.set_position(sp)
        self.other_agent.set_position(op)

    def get_agent_pos(self):
        return self.self_agent.get_position(), self.other_agent.get_position()

    def off_display(self):
        self.world.off_display()

    def capture(self):
        self.world.draw(self.self_agent, self.other_agent)
        return self.world.capture()

    def capture_at_pos(self, sp, op):
        self.set_agent_pos(sp, op)
        return self.capture()

    def set_camera(self, key):
        self.world.set_camera(key)

    def step(self):
        v = self.capture()

        sp = copy.deepcopy(self.self_agent.p)
        op = copy.deepcopy(self.other_agent.p)

        sm = self.self_agent.step()
        om = self.other_agent.step()

        return v, sm, om, sp, op
