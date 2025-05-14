import json
import numpy as np
from openai import OpenAI
from state import State

class DecideDataset(object):
    def __init__(self, llm, project_info):
        self.llm = llm
        self.project_info = project_info
        self.sampled_datasets = project_info.get_sampled_dataset_dict()
        self.final_flow_structure = project_info.create_final_flow_structure()
        self.relevant_facets = project_info.get_relevant_facets()

    def decide_dataset(self, state: State):
        # THIS is a deterministic way using semantic vectorizations to decide the dataset
        if state["verbose"]:
            print("Finding a dataset to answer your question...")
        # Just use self.project_info.index and vectorize query
        client = OpenAI(api_key=self.project_info.openai_api_key)
        # Also incorporate previous information here
        response = client.embeddings.create(
            input=f"previous history: {state['previous_info']}. new query: {state['user_query']}",
            model="text-embedding-3-small",
        )
        result = self.project_info.index.search(np.array([response.data[0].embedding]).astype(np.float32), 1)[1]
        # Find dataset name corresponding with index returned
        # Returned index is a 2d array with just one number
        dataset_name = self.project_info.dataset_names_for_vectorizations[result[0][0]]
        if state["verbose"]:
            print(f"Selected dataset: {dataset_name}")
        return {"dataset_name": dataset_name}

    def decide_dataset_llm(self, state: State):
        # THIS is a more robust method for finding dataset using better information
        # It is less deterministic, but often works more robustly
        if state["verbose"]:
            print("Finding a dataset to answer your question...")
        FIRST_DECIDE_DATASET_PROMPT = f"""
            You will be given a list of datasets. 
            For each dataset you will be provided with a sentence long description.

            You will also be given a json that explains the way these datasets are all connected.
            Each key is the name of the flow (or dataset connection) and each value is a list of 
            operations that happen inside of the flow. Often, the first action imports a specific dataset
            in some way. Then the final action exports a new dataset. Both of which exist in the aformentioned
            list of datasets Each item in the list of operations has an 'id' which is a unique identifier for the
            operation. It has a 'type' which explains which operation is being done. It has 'inputs' which is a list 
            of the ids of the inputs to that operation. It has a 'dataset_name' which is the name of the input or 
            output dataset that is directly associated with that operation. They exist mostly for import and export
            data operations.

            You will be given a list of descriptions of the operations that are listed under 'type' in 
            the json. These will provide further context on exactly what is being done to the dataset.

            Understand the relationships between the datasets. Given previous information about past 
            user queries + responses and a new user query, be prepared to answer which
            dataset would contain the answer to this new user query.

            List of datasets
            {self.sampled_datasets}

            Json explaining dataset connections:
            {self.final_flow_structure}

            Descriptions of operations:
            {self.relevant_facets}

            Remember, respond with the dataset that would contain the answer the user query.
            Consider carefully how the datasets are used in the context of the json explaining connections.
            If a dataset is not connected to others, it is less likely to be used.
            The names of the datasets are less important that the json explaining connections
            and the descriptions of operations. Select a dataset considering all of these factors
            and explain why. Consider the previous info, particularly
            the dataset that was used to generate the first response. It will likely be the same dataset,
            although potentially it may be different as well. If you explain why and need to select a different dataset,
            please do so. Return a json of the form 'response': 'insert explanation here'.
        """
        # Get answer from LLM
        messages = [
            ("system", FIRST_DECIDE_DATASET_PROMPT), 
            ("human", f"Previous info: {state['previous_info']}"),
            ("human", f"User query: {state['user_query']}")
        ]
        answer = self.llm.invoke(messages).content
        messages.append(('assistant', answer))
        SECOND_DECIDE_DATASET_PROMPT = f"""
            Now that you understand general structure of the datasets and a potential choice.
            Rethink this choice again carefully in the context of all the datasets, and all the 
            connections/operations between these datasets. Consider the previous info, particularly
            the dataset that was used to generate the first response. It will likely be the same dataset,
            although potentially it may be different as well.
            Choose the name of dataset from this list: {self.sampled_datasets.keys()}
            Return a json in the following structure. {{'dataset': 'name of dataset'}}
        """
        messages.append(('human', SECOND_DECIDE_DATASET_PROMPT))
        answer = self.llm.invoke(messages).content
        try:
            dataset_name = self.get_dataset_name(answer)
        except:
            if state["verbose"]:
                print("First try didn't work for dataset decision")
            messages.append(("assistant", answer))
            instructions = f"Make sure you reply with a json in the format {{'dataset': 'name of dataset'}} where 'name of dataset' is in the list {self.sampled_datasets.keys()}."
            messages.append(("human", instructions))
            answer = self.llm.invoke(messages).content
            try:
                dataset_name = self.get_dataset_name(answer)
            except:
                messages.append(("assistant", answer))
                THIRD_DECIDE_DATASET_PROMPT = f"""
                    Can you instead provide me with 3 sentences of relevant information 
                    based on all the information you have to the user query: {state["user_query"]}. 
                    Describe which datasets might be helpful or things the user might want to look
                    into.
                """
                messages.append(("human", THIRD_DECIDE_DATASET_PROMPT))
                answer = self.llm.invoke(messages).content
                # If it doesn't work the second time, update validity to false
                return {"is_valid": False, "helpful_info": answer}
        if state["verbose"]:
            print(f"Selected dataset: {dataset_name}")
        return {"dataset_name": dataset_name}

    def get_dataset_name(self, answer):
        dataset_name = json.loads(answer)['dataset']
        if dataset_name in self.sampled_datasets.keys():
            return dataset_name
        else:
            raise ValueError