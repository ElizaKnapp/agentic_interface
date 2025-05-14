from state import State
import json

ARBITRARY_CATEGORY_CAP = 20
ARBITRARY_SAMPLE_LENGTH = 5

class DecideFilters(object):
    def __init__(self, llm, project_info):
        """
        llm 
        """
        self.llm = llm
        self.project_info = project_info
    
    def decide_filters(self, state: State):
        if state["verbose"]:
            print("Filtering dataset based on your question...")
        # Get dataset
        dataset_name = state["dataset_name"]
        dataset_object = self.project_info.get_dataset_object(dataset_name)
        # Get datatypes and categorical columns
        datatypes = dataset_object.data_types
        # Get sample
        df = dataset_object.df()
        df_sample = df.sample(n=min(ARBITRARY_SAMPLE_LENGTH,len(df.index)))
        categorical_columns = []
        # Also add text columns because datatype detection is weird...
        for column_name, column_datatype in datatypes.items():
            if column_datatype.data_type == "CATEGORICAL" or column_datatype.data_type == 'TEXT':
                if len(df[column_name].unique()) < ARBITRARY_CATEGORY_CAP:
                    categorical_columns.append(column_name)
        # Get unique values for categorical columns
        unique_values_dict = dict()
        for column in categorical_columns:
            unique_values_dict[column] = list(df[column].unique())

        DECIDE_FILTERS_PROMPT = f"""
            You will be given dataset information, previous user queries and responses, and a new user query. 
            Here is the dataset information.
            Dataset: {df_sample}
            Based on the new user query, you need to determine which categorical columns to filter.
            I am only looking for filters on the categorical columns: {categorical_columns}.
            For each of those columns, here is a list of unique values to potentially
            filter on.
            Unique values dict: {unique_values_dict}
            You need to take in the new user query, and decide which of the categorical columns need to 
            be filtered. Return the output in the following json format:
            This is an example, replace the keys and values in this dictionary.
            {{
                "insert name of column": ["filter1", "filter2"]
            }}
            Remember, the filters can only come from the values in Unique values dict for a specific
            categorical column name.
            All the dictionary keys should belong to this list: {categorical_columns}
            If "prediction_type" is one of the columns, filter for the "predicted_value" strings in that
            column to avoid the real values/lower and upper bounds messing up metrics.
        """
        messages = [
            ('system', DECIDE_FILTERS_PROMPT), 
            ("human", f"Previous info: {state['previous_info']}"),
            ('human', f"User query: {state['user_query']}")
        ]
        answer = self.llm.invoke(messages).content
        try:
            llm_filters = json.loads(answer)
            # Check the values
            final_filters = dict()
            for column in categorical_columns:
                if column not in llm_filters.keys():
                    # If the column isn't there, don't filter
                    final_filters[column] = None
                else:
                    # Otherwise, make sure the filters are in the columns
                    refined_filters = list(llm_filters[column].copy())
                    for value in llm_filters[column]:
                        if value not in unique_values_dict[column]:
                            refined_filters.remove(value)
                    final_filters[column] = refined_filters    
        except:
            return {"is_filter_valid": False, "filter_attempts": state["filter_attempts"] + 1, "helpful_info": None}
        return {"is_filter_valid": True, "filters": final_filters, "filter_attempts": state["filter_attempts"] + 1}
