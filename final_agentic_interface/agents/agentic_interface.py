import pandas as pd
from langgraph.graph import StateGraph, START, END
from state import State
from IPython.display import Image, display
# Nodes
from validate_prompt import ValidatePrompt
from decide_dataset import DecideDataset
from relevant_info import RelevantInfo
from dataset_descriptor import DatasetDescriptor
from decide_filters import DecideFilters
from write_code import WriteCode
from format_response import FormatResponse
# Edges
from edges import Edges

class AgenticInterface(object):
    """
    This class handles the creation and asking questions to the
    agentic interface
    """
    def __init__(self, llm, json_llm, project_info, verbose=True):
        # Set the verbosity
        self.verbose = verbose
        # Initialize each of the agents
        validatePrompt = ValidatePrompt(llm=llm, project_info=project_info)
        decideDataset = DecideDataset(llm=json_llm, project_info=project_info)
        datasetDescriptor = DatasetDescriptor(llm=llm, project_info=project_info)
        decideFilters = DecideFilters(llm=json_llm, project_info=project_info)
        writeCode = WriteCode(llm=llm, project_info=project_info)
        formatResponse = FormatResponse(llm=llm, project_info=project_info)
        relevantInfo = RelevantInfo(llm=llm)
        # Initialize the edge class
        edges = Edges(project_info=project_info)

        # Begin the graph with a predefined state
        workflow = StateGraph(State)
        # Add nodes
        workflow.add_node("validate_prompt", validatePrompt.validate_prompt)
        workflow.add_node("decide_dataset", decideDataset.decide_dataset)
        workflow.add_node("relevant_info", relevantInfo.relevant_info)
        workflow.add_node("dataset_descriptor", datasetDescriptor.dataset_descriptor)
        workflow.add_node("decide_filters", decideFilters.decide_filters)
        workflow.add_node("write_code", writeCode.write_code)
        workflow.add_node("format_response", formatResponse.format_response)

        # Add edges
        workflow.add_edge(START, "validate_prompt")
        workflow.add_conditional_edges(
            "validate_prompt", edges.check_validity, {True: "decide_dataset", False: END}
        )
        workflow.add_conditional_edges(
            "decide_dataset", edges.check_validity, {True: "dataset_descriptor", False: "relevant_info"}
        )
        workflow.add_edge("dataset_descriptor", "decide_filters")
        workflow.add_conditional_edges("decide_filters", edges.check_filters, {"Exit": "relevant_info", "Pass": "write_code", "Fail": "decide_filters"})
        workflow.add_conditional_edges("write_code", edges.check_code, {"Exit": "relevant_info", "Pass": "format_response", "Fail": "write_code"})
        workflow.add_conditional_edges("format_response", edges.check_validity, {True: END, False: "relevant_info"})
        workflow.add_edge("relevant_info", END)

        # Compile the workflow
        self.chain = workflow.compile()

        # Define current state
        self.current_state = None

        # Define saved chat
        self.chat_history = {
            "Time": [],
            "User Email": [],
            "User Query": [],
            "Response": []
        }

        # Define the previous info that we will append to for the chat
        self.previous_info = dict()
        self.current_chat_number = 1
    
    def get_chain(self):
        return self.chain
    
    def visualize_chain(self):
        display(Image(self.chain.get_graph().draw_mermaid_png()))
    
    def get_current_state(self):
        return self.current_state
    
    def write_current_state_to_general_logs(self, destination):
        with open(f"{destination}", 'a') as f:
            f.write(str(self.current_state) + '\n')
        
    def upload_chat_history_to_file(self, destination):
        """
        Filename is just the name without the extension of the file
        """
        chat_history_df = pd.DataFrame(self.chat_history)
        chat_history_df.to_csv(f"{destination}", index=False)

    def get_response(self, user_query):
        state_input_dict = {
            "user_query": user_query, 
            "time": None,
            "user_email": None,
            "is_valid": True,
            "dataset_name": None,
            "dataset_description": None,
            "filters": None,
            "filter_attempts": 0, 
            "is_filter_valid": True,
            "generated_code": None,
            "code_attempts": 0, 
            "generated_code_output": None,
            "response": None,
            "helpful_info": None,
            "previous_info": self.previous_info, 
            "verbose": self.verbose
        }
        self.current_state = self.chain.invoke(state_input_dict)
        # Add to previous info
        chat_summary_string = f"""
            User query: {self.current_state["user_query"]}
            Dataset: {self.current_state["dataset_name"]}
            Dataset Description: {self.current_state["dataset_description"]}
            Agent response: {self.current_state["response"]}
        """
        self.previous_info[f"Chat {self.current_chat_number}"] = chat_summary_string
        # Change the chat history
        self.chat_history["Time"].append(self.current_state["time"])
        self.chat_history["User Email"].append(self.current_state["user_email"])
        self.chat_history["User Query"].append(self.current_state["user_query"])
        self.chat_history["Response"].append(self.current_state["response"])
        # Update the current chat number
        self.current_chat_number += 1
        # Return the response only
        return self.current_state["response"]

    