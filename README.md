# agentic_interface
Final neuro 240 project

Inside of the final_agentic_interface folder
- agents contains the bulk of the code
- data is Ikigai platform specific information that is helpful for my agents
- prototyping is my initial jupyter notebook prototype before I improved and modularized it

Inside of the final_agentic_interface/agents folder
- chat_history is where all of the chat history dataframes from my testing are saved
- result_files are results I gathered including my routine log file to improve code via experimentation
- results_for_paper are the results that I used along with calculate_metrics.ipynb to create all the figures in the paper
- agentic_interface.py contains the class that uses langgraph to coordinate all the nodes and edges
- calculate_metrics.ipynb has metrics for the paper
- dataset_descriptor.py takes in a dataset sample and describes it
- decide_dataset.py has two methods for deciding datasets
- decide_filters.py decides which filters are used based on the query
- driver.ipynb contains an example of the agentic interface being used
- edges.py defines the transitions for the langgraph function
- format_response.py contains the conversion from code output to question answer
- project_info.py contains the coordinations for all the relevant Ikigai data pulling that is vital to the agentic interface
- relevant_info.py provides relevant information when the query errors out or the question cannot be answered
- state.py is a reminder of the variables maintained throughout the langgraph state
- uid_to_name.csv contains an Ikigai specific mapping for function names
- validate_prompt.py decides whether the prompt can be answered by the project
- write_code.py writes the code to answer the query based on a filter and the dataset
