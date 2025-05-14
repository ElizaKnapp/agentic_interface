import ast
from state import State

ARBITRARY_SAMPLE_LENGTH = 20

class WriteCode(object):
    def __init__(self, llm, project_info):
        """
        llm 
        """
        self.llm = llm
        self.project_info = project_info
    
    def write_code(self, state: State):
        if state["verbose"]:
            print("Generating results based on your query...")
        # Get dataset
        dataset_name = state["dataset_name"]
        dataset_object = self.project_info.get_dataset_object(dataset_name)
        # Get datatypes
        datatypes = dataset_object.data_types
        # Get sample
        df = dataset_object.df()
        df_sample = df.sample(n=min(ARBITRARY_SAMPLE_LENGTH,len(df.index)))
        # Setup prompt
        WRITE_CODE_PROMPT = f"""
            You will be given a user query, previous queries and responses, a sample of a pandas dataframe and datatype information,
            and suggested filters on some of the columns.
            Write code in python that takes in a dataframe and outputs a dictionary of numeric values 
            that are helpful in answering the user query. Also take the previous queries (if they exist) into account
            while understanding exactly what the user query is asking. The code should filter columns
            according to the following filters as needed: {state["filters"]}. There should be no
            additional filtering of the text or categorical columns.
            Also, if "prediction_type" is one of the columns, filter for the "predicted_value" rows
            and "real" rows and only compute metrics on one of them at a time.
            The output should also contain some additional relevant details. 
            Ensure that you are careful of datatypes when writing code. 
            Dataframe: {df_sample}
            Datatype Information: {datatypes}
            - Note that even though some columns have time datatype, you must still 
            convert them explicitly in the code to pd.datetime objects before
            doing operations on them. Assume they are just objects until code is run
            on them.
            Return a function with appropriate docstring in the following format:
                ```python
                    {{function body}}
                ```
            Only return one python function. No testing or additional information outside
            of the above specified format. Import all required libraries inside of the 
            function definition.
        """
        messages = [
            ("system", WRITE_CODE_PROMPT), 
            ("human", f"Previous info: {state['previous_info']}"),
            ("human", f"User query: {state['user_query']}.")
        ]
        answer = self.llm.invoke(messages).content
        try:
            generated_code = answer.strip().removeprefix("```python\n").removesuffix("\n```")
            exec_env = {}
            exec(generated_code, exec_env, exec_env)
            if state["verbose"]:
                print("Finished generating code")
            function_name = self.extract_function_names(generated_code)[0]
            function_output = exec_env[function_name]
            output = function_output(df)
            if state["verbose"]:
                print("Finished running function on data")
            return {"is_code_valid": True, "code_attempts": state["code_attempts"] + 1,"generated_code_output": output, "generated_code": generated_code}
        except:
            if state["verbose"]:
                print("Function generation didn't work")
                print(answer)
            return {"is_code_valid": False, "code_attempts": state["code_attempts"] + 1, "helpful_info": None}

    def extract_function_names(self, code_str):
        """
        Parses a Python code string and returns a list of function names defined in it.
        """
        tree = ast.parse(code_str)
        return [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]