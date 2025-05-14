import requests
import faiss
import numpy as np
from ikigai import Ikigai
from openai import OpenAI

INDEX_DIMENSION = 1536 # because openai

class ProjectInfo(object):
    """
    Class for handling all interactions with an ikigai project
    """

    def __init__(self, user_email, api_key, openai_api_key, target_project, mapping_df):
        """
        user_email (str)
        api_key (str): ikigai api key
        openai_api_key (str): openai api key
        target_project (str): the name of the project
        mapping_df (pandas df): in the future, won't be needed but now 
        contains UID to facet name mapping
        """
        # Pull target project
        self.user_email = user_email
        ikigai = Ikigai(user_email=self.user_email, api_key=api_key)
        apps = ikigai.apps()
        target_app = apps[target_project]
        # Save the datasets/flows
        self.datasets = target_app.datasets()
        self.flows = target_app.flows()
        # Save all the vectors of a dataset
        client = OpenAI(api_key=openai_api_key)
        self.openai_api_key = openai_api_key
        self.vectorizations = []
        self.dataset_names_for_vectorizations = []
        self.index = faiss.IndexFlatIP(INDEX_DIMENSION)

        # Setup the sampled dataset dict
        self.n_samples = 2
        self.sampled_dataset_dict = dict()
        for dataset_name in self.datasets:
            df = self.datasets[dataset_name].df()
            self.sampled_dataset_dict[dataset_name] = df.sample(n=min(self.n_samples,len(df.index)), random_state=42)
            # Add the vectors to the list
            self.dataset_names_for_vectorizations.append(dataset_name)
            # Get the actual embeddings
            response = client.embeddings.create(
                input=f"dataset name: {dataset_name}. dataset_sample: {self.sampled_dataset_dict[dataset_name].to_string(index=False)}",
                model="text-embedding-3-small",
            )
            self.vectorizations.append(response.data[0].embedding)

        # Upsert all the vecotrs to the index
        payload = np.array(self.vectorizations).astype(np.float32)
        self.index.add(payload)

        self.sampled_dataset_dict = self.sampled_dataset_dict
        # Setup the pipeline definitions
        BASE_URL = "https://api.ikigailabs.io/"
        headers = {
                    'User': self.user_email,
                    'Api-key': api_key,
                    'Content-Type': 'application/json'
                }
        self.pipeline_definitions = dict()
        for flow in self.flows:
            pipeline_id = self.flows[flow].flow_id
            url = f"{BASE_URL}/component/get-pipeline?project_id={target_app.app_id}&pipeline_id={pipeline_id}"
            pipeline_response = requests.request("GET", url, headers=headers)
            if pipeline_response.status_code!=200:
                print("Error getting pipeline")
            else:
                pipeline_definition = pipeline_response.json()
                self.pipeline_definitions[flow] = pipeline_definition
        # Setup UID to facet mapping
        self.uid_to_name = dict()
        for i in range(len(mapping_df.index)):
            row = mapping_df.iloc[i]
            self.uid_to_name[row["UID"]] = row["Facet Name"]
        self.facet_to_description = dict()
        for i in range(len(mapping_df.index)):
            row = mapping_df.iloc[i]
            self.facet_to_description[row["Facet Name"]] = row["Description"]

    def get_sampled_dataset_dict(self):
        return self.sampled_dataset_dict
    
    def get_pipeline_definitions(self):
        return self.pipeline_definitions

    def get_dataset_object(self, dataset_name):
        return self.datasets[dataset_name]

    def get_dataset(self, dataset_name):
        return self.datasets[dataset_name].df()

    def get_uid_to_name(self):
        return self.uid_to_name

    def get_facet_to_description(self):
        return self.facet_to_description
    
    def parse_pipeline_to_flat_json(self, pipeline_json):
        nodes = []
        inputs_map = dict()  # maps destination facet to list of source facets

        # Build a mapping of destination to sources
        for arrow in pipeline_json['pipeline']['definition']['arrows']:
            source = f"Facet {arrow['source']}"
            dest = f"Facet {arrow['destination']}"
            inputs_map.setdefault(dest, []).append(source)

        # Parse each facet into a node
        for facet in pipeline_json['pipeline']['definition']['facets']:
            facet_id = f"Facet {facet['facet_id']}"
            facet_type = self.uid_to_name.get(facet['facet_uid'], 'Unknown')
            dataset_name = facet['arguments'].get('dataset_name', facet['name'])
            node = {"id": facet_id, "type": facet_type}
            if facet_id in inputs_map:
                node["inputs"] = inputs_map[facet_id]
            if dataset_name:
                node["dataset_name"] = dataset_name
            nodes.append(node)
        return nodes

    def create_final_flow_structure(self):
        self.final_flow_structure = dict()
        # Go through each pipeline definition and make a json of the name -> json
        for name, pipeline_definition in self.pipeline_definitions.items():
            self.final_flow_structure[name] = self.parse_pipeline_to_flat_json(pipeline_definition)
    
    def get_relevant_facets(self):
        # First get unique facets used
        facets = set()
        for item in self.final_flow_structure.values():
            for value in item:
                facets.add(value['type'])
        # Then create mapping
        relevant_facets = dict()
        for value in facets:
            relevant_facets[value] = self.facet_to_description[value]
        return relevant_facets
    
    def get_user_email(self):
        return self.user_email