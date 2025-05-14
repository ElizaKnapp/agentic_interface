from state import State


class RelevantInfo(object):
    def __init__(self, llm):
        """
        llm 
        """
        self.llm = llm
    
    def relevant_info(self, state: State):
        if state["verbose"]:
            print("Direct answer to your question could not be found. Generating relevant information...")
        if state["helpful_info"]:
            relevant_info = state["helpful_info"]
        else:
            RELEVANT_INFO_PROMPT = f"""
                Given a user query, previous queries and responses, and a dataset name and description, 
                provide some helpful information that is related tangentially 
                to the user query,
                Dataset name: {state["dataset_name"]}
                Dataset description: {state["dataset_description"]}
            """
            messages = [
                ("system", RELEVANT_INFO_PROMPT), 
                ("human", f"Previous info: {state['previous_info']}"),
                ("human", f"User query: {state['user_query']}")]
            relevant_info = self.llm.invoke(messages).content
        response = f"""
            Sorry we could not answer the query. Here is some relevant information
            that might help you get started. {relevant_info}
        """
        if state["verbose"]:
            print("Relevant information generated")
        return {"response": response}