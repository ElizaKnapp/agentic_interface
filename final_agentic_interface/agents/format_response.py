from state import State

ARBITRARY_SAMPLE_LENGTH = 5

class FormatResponse(object):
    def __init__(self, llm, project_info):
        """
        llm 
        """
        self.llm = llm
        self.project_info = project_info
    
    def format_response(self, state: State):
        if state["verbose"]:
            print("Finalizing response to your question...")
        # Get dataset
        dataset_name = state["dataset_name"]
        df = self.project_info.get_dataset(dataset_name)
        df_sample = df.sample(n=min(ARBITRARY_SAMPLE_LENGTH,len(df.index)))

        RESPONSE_PROMPT = f"""
            You will be given a dataframe, code that runs on this dataframe,
            and the output of that code. You will use this output and information
            about what the code does to answer the user query.
            Name of dataframe: {state["dataset_name"]}
            Dataframe sample: {df_sample}
            Code: {state['generated_code']}
            Output of code: {state['generated_code_output']}
            Answer the user query with an explanation of how the numbers were generated
            from which dataset. Answer as if you are just answering the user query as a 
            chatbot, but make sure to use the necessary information. Use all the columns
            of the dataframe to explain what the numbers mean in the context of the data.
            For example, if you are asked to find a metric, tell us what the name of that 
            metric is if it is available in the dataset.
            If it is helpful, you also can use the previous info to contextualize the question
            and provide comparisons and additional info where it is useful.
            If the data cannot answer the user query, answer with just the word 'No'.
        """
        messages = [
            ("system", RESPONSE_PROMPT),  
            ("human", f"Previous info: {state['previous_info']}"),
            ("human", f"User query: {state['user_query']}")]
        answer = self.llm.invoke(messages).content
        if answer == 'No':
            return {"is_valid": False}
        else:
            if state["verbose"]:
                print("Response finalized")
            return {"response": answer, "is_valid": True}