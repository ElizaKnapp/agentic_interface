from typing_extensions import TypedDict

class State(TypedDict):
    user_query: str
    time: str
    user_email: str
    is_valid: bool
    dataset_name: str
    dataset_description: str
    filters: dict
    filter_attempts: int
    is_filter_valid: bool
    generated_code: str
    code_attempts: int
    is_code_valid: bool
    generated_code_output: dict
    response: str
    helpful_info: str
    previous_info: dict
    verbose: bool