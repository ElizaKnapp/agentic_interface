from datetime import datetime
from state import State


class ValidatePrompt(object):
    def __init__(self, llm, project_info):
        """
        llm 
        sampled_datasets (dict of dfs)
        """
        self.llm = llm
        self.project_info = project_info
        self.sampled_datasets = self.project_info.get_sampled_dataset_dict()

    def validate_prompt(self, state: State):
        if state["verbose"]:
            print(f"You asked: {state['user_query']}")
            print("Validating your query...")
        # Setup the system context
        VALIDATE_QUERY_PROMPT = f"""
            You will be given a dictionary of datasets and their descriptions in a project and a user query.
            Before the user query, you will also be given the current chat history if it exists.
            Datasets:
            {self.sampled_datasets}
            If the question provided is even tangentially related to the dictionary of 
            datasets above, return just 'Yes', Otherwise, return 'No'. 
            A 'No' result would only be if the question seems unrelated to 
            the datasets above.
            Assume that in general, prompts that have to do with forecasting data are 
            generally valid.
        """
        # Get answer from LLM
        messages = [
            ("system", VALIDATE_QUERY_PROMPT), 
            ("human", f"Previous history: {state['previous_info']}"),
            ("human", f"User query: {state['user_query']}")
        ]
        answer = self.llm.invoke(messages).content
        # Filter that answer to a boolean value
        if answer == "Yes":
            is_valid = True
        elif answer == "No":
            is_valid = False
        else:
            if state["verbose"]:
                print("First output of prompt validation was not yes or no")
            # Otherwise specify that llm must return yes or no
            messages.append(("assistant", answer))
            messages.append(("human", "Respond with just 'Yes' or 'No' to whether the user query is related to the datasets."))
            answer = self.llm.invoke(messages).content
            # If it still won't answer with one word, just pass it on to next step
            is_valid = False if answer == "No" else True
        # Since this happens first, also save the user email and time
        timestamp_str = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        final_state = {"is_valid": is_valid, "user_email": self.project_info.get_user_email(), "time": timestamp_str}
        if not is_valid:
            final_state["response"] = "Sorry, we are unable to answer your question given this project data."
        else:
            if state["verbose"]:
                print("Query is valid")
        return final_state
    
