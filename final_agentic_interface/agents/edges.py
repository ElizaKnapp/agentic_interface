from state import State

ARBITRARY_FILTER_ATTEMPT_CAP = 5
ARBITRARY_CODE_ATTEMPT_CAP = 3

class Edges(object):
    def __init__(self, project_info):
        self.project_info = project_info

    def check_validity(self, state: State):
        return state["is_valid"]

    def check_filters(self, state: State):
        if not state["is_filter_valid"] and state["filter_attempts"] >= ARBITRARY_FILTER_ATTEMPT_CAP:
            return "Exit"
        elif not state["is_filter_valid"]:
            return "Fail"
        else:
            return "Pass"
    
    def check_code(self, state: State):
        if not state["is_code_valid"] and state["code_attempts"] >= ARBITRARY_CODE_ATTEMPT_CAP:
            return "Exit"
        elif not state["is_code_valid"]:
            return "Fail"
        else:
            return "Pass"

        