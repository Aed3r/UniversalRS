from SPARQLWrapper import SPARQLWrapper, JSON, TURTLE
import pandas as pd
import requests
from tqdm import tqdm

DATASETS = {
    'Movielens1M': './data/LODrecsys-datasets/Movielens1M/MappingMovielens2DBpedia-1.2.tsv',
    'Librarything': './data/LODrecsys-datasets/Librarything/MappingLibrarything2DBpedia-1.2.tsv',
    'Lastfm': './data/LODrecsys-datasets/LastFM/MappingLastfm2DBpedia-1.2.tsv',
}

class SPARQLEndpoint:
    def __init__(self, endpoint_url):
        self.endpoint_url = endpoint_url
        self.sparql = None
        self.lookup_url = "https://lookup.dbpedia.org/api/search"

    def connect(self):
        """
        Initializes connection
        """
        self.sparql = SPARQLWrapper(self.endpoint_url)

    def disconnect(self):
        """
        Disconnects from endpoint
        """
        self.sparql = None

    def query(self, query_string):
        """
        Executes a query against the SPARQL endpoint and returns the results in JSON format.
        """
        if not self.sparql:
            raise Exception("Not connected to any SPARQL endpoint.")

        self.sparql.setQuery(query_string)
        self.sparql.setReturnFormat(JSON)
        
        return self.sparql.query().convert()
    
    def lookup_term(self, term, field="query", exact=False):
        """
        Lookup a term using the DBpedia Lookup service to get the most likely URI.

        Args:
            term (str): The term to lookup.
            field (str): The field to search on. Default is "query" which searches on all default fields.
            exact (bool): Whether to enforce an exact match on the field.

        Returns:
            str: The most likely DBpedia URI for the term, or None if no match found.
        """
        
        params = {
            field: term,
            "maxResults": 1,
            "format": "JSON_RAW"
        }

        if exact:
            params[field + "Exact"] = "true"

        response = requests.get(self.lookup_url, params=params)

        if response.status_code == 200:
            data = response.json()
            if data['docs'] and len(data['docs']) > 0:
                return data['docs'][0]['resource'][0]
        return None
    
    def describe_resource(self, resource_uri):
        """
        Retrieve the full DBpedia document for a given resource URI.

        Args:
            resource_uri (str): The URI of the resource to describe.

        Returns:
            str: A string representation of the RDF graph for the resource in Turtle format.
        """
        # Construct a SPARQL DESCRIBE query for the given resource URI.
        query_string = f"DESCRIBE <{resource_uri}>"
        
        self.sparql.setQuery(query_string)
        self.sparql.setReturnFormat(TURTLE)

        try:
            results = self.sparql.query().convert()
            return results
        except Exception as e:
            print(f"Error querying the SPARQL endpoint: {e}")
            return None

def load_mapping(dataset):
    if dataset not in DATASETS.keys():
        print("Invalid dataset specified.")
        exit()
    
    print(f"Loading {dataset} mapping...")
    mapping = pd.read_csv(DATASETS[dataset], sep='\t', names=['ItemID', 'Title', 'ManualDBpediaURI'])
    print(f"Loaded {len(mapping)} item mappings.")

    return mapping

if __name__ == "__main__":
    endpoint = SPARQLEndpoint("http://dbpedia.org/sparql")
    #endpoint.connect()

    # Load the mapping
    mapping = load_mapping('Lastfm') # Movielens1M, Librarything, Lastfm

    # Lookup all terms in the mapping and get the most likely DBpedia URI.
    for index, row in tqdm(mapping.iterrows(), total=len(mapping), desc="Creating mappings"):
        uri = endpoint.lookup_term(row['Title'])
        mapping.at[index, 'AutoDBpediaURI'] = uri

    # Compare the manual and automatic mappings. Count correct mappings, incorrect mappings, and missing mappings.
    correct = 0
    incorrect = 0
    missing = 0
    for index, row in tqdm(mapping.iterrows(), total=len(mapping), desc="Comparing mappings"):
        if row['ManualDBpediaURI'] == row['AutoDBpediaURI']:
            correct += 1
        elif row['ManualDBpediaURI'] and not row['AutoDBpediaURI']:
            missing += 1
        else:
            incorrect += 1
    
    print(f"Correct mappings: {correct}")
    print(f"Incorrect mappings: {incorrect}")
    print(f"Missing mappings: {missing}")

    # Disconnect from the endpoint.
    endpoint.disconnect()