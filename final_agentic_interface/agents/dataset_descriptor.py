from state import State


class DatasetDescriptor(object):
    def __init__(self, llm, project_info):
        """
        llm 
        """
        self.llm = llm
        self.project_info = project_info
    
    def dataset_descriptor(self, state: State):
        # Get dataset
        dataset_name = state["dataset_name"]
        dataset_object = self.project_info.get_dataset_object(dataset_name)
        # Get datatypes
        datatypes = dataset_object.data_types
        # Get sample
        df = dataset_object.df()
        df_sample = df.sample(n=min(20,len(df.index)))
        # Setup prompt
        DATASET_DESCRIPTOR_PROMPT = f"""
            You will be given a sample of a dataset and the datatypes of each of the columns.
            Dataset: {df_sample}
            Column data types: {datatypes}
            Return a couple sentences that describe the type of information this dataset contains.
        """
        # Get answer from LLM
        answer = self.llm.invoke(DATASET_DESCRIPTOR_PROMPT).content
        if state["verbose"]:
            print(f"Here is the description for your dataset: {answer}")
        return {"dataset_description": answer}