from abc import ABCMeta, abstractmethod


class RNNMeta(metaclass=ABCMeta):
    @abstractmethod
    def init_state(self, batch_size):
        pass

    @abstractmethod
    def detach_state(self):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def get_detached_state(self):
        pass


class RNNBase(RNNMeta):
    def detach_state(self):
        self.state = self.get_detached_state()

    def get_state(self):
        _state = {}
        for k in self.state.keys():
            _state[k] = self.state[k]
        return _state

    def get_detached_state(self):
        _state = {}
        for k in self.state.keys():
            _state[k] = self.state[k]._replace(
                **{
                    name: value.detach()
                    for name, value in self.state[k]._asdict().items()
                })
        return _state


class WithRNNModuleBase(RNNMeta):
    def init_state(self, batch_size):
        for rnn in self.rnn_modules:
            rnn.init_state(batch_size)

    def detach_state(self):
        for rnn in self.rnn_modules:
            rnn.detach_state()

    def get_state(self):
        _state = {}
        for rnn in self.rnn_modules:
            _state.update(rnn.get_state())
        return _state

    def get_detached_state(self):
        _state = {}
        for rnn in self.rnn_modules:
            _state.update(rnn.get_detached_state())
        return _state
