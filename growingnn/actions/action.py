
from growingnn.actions.utils.layer_Factory import Layer_Type
from growingnn.core.traced_model import TracedModel


class Action:
    def __init__(self, _params):
        self.params = _params

    def execute(self, traced: TracedModel) -> None:
        self._execute(traced)
        traced.invalidate()

    def _execute(self, traced: TracedModel) -> None:
        pass

    def can_be_infulenced(self, by_action):
        pass

    @staticmethod
    def generate_all_actions(model, grow: bool = True, shrink: bool = True):
        return []
