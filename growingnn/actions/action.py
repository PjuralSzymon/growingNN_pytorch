

from growingnn.actions.utils.layer_Factory import Layer_Type


class Action:
    def __init__(self, _params):
        self.params = _params
        pass

    def execute(self, model):
        pass

    def can_be_infulenced(self, by_action):
        pass

    @staticmethod
    def generate_all_actions(model, grow: bool = True, shrink: bool = True):
        from growingnn.actions.registry import generate_all_actions as registry_generate

        return registry_generate(model, grow=grow, shrink=shrink)