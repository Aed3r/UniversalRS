from SPARQLWrapper import SPARQLWrapper, JSON, TURTLE
import requests
import json
from rdflib import Graph, URIRef

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
        
    def write_resource_to_file(self, resource_uri, filename):
        """
        Writes the RDF description of a resource to a file.

        Args:
            resource_uri (str): The URI of the resource to describe.
            filename (str): The name of the file to write to.
        """
        rdf_data = self.describe_resource(resource_uri)
        if rdf_data:
            with open(filename, 'wb') as file:
                file.write(rdf_data)
        else:
            print(f"Failed to retrieve data for {resource_uri}.")

    def get_related_items(self, uri, limit=10, offset=0):
        """
        Retrieves items related to a given URI from the LOD graph with pagination.

        Args:
            uri (str): The URI of the item to find related items for.
            limit (int): The maximum number of related items to retrieve per page.
            offset (int): The starting point from which to retrieve the results.

        Returns:
            list: A list of URIs of related items.
        """
        query = f"""
            SELECT DISTINCT ?related WHERE {{
                {{ ?related ?p1 <{uri}> }} UNION  # Subjects related to the URI
                {{ <{uri}> ?p2 ?related . FILTER(!isLiteral(?related)) }}  # Non-literal objects related to the URI
            }} LIMIT {limit} OFFSET {offset}
        """
        
        results = self.query(query)
        if not results:
            return []

        related_items = []
        for result in results["results"]["bindings"]:
            related_items.append(result["related"]["value"])

        return related_items
    
    @staticmethod
    def filter_rdf_triples(rdf_data, format, allowed_predicates):
        """
        Filters the RDF triples of a given RDF data based on a list of allowed predicates.

        Args:
            rdf_data (str): RDF data as a string.
            format (str): The format of the RDF data (e.g., 'turtle', 'xml', 'n3', etc.).
            allowed_predicates (list): List of URIs of allowed predicates.

        Returns:
            str: Filtered RDF data as a string in the same format as the input.
        """
        # Parse the original RDF data
        g = Graph()
        g.parse(data=rdf_data, format=format)

        # Create a new graph to store the filtered triples
        filtered_g = Graph()

        # Filter the triples
        for s, p, o in g:
            if str(p) in allowed_predicates:
                filtered_g.add((s, p, o))

        # Serialize the filtered graph to a string
        filtered_rdf_data = filtered_g.serialize(format=format, encoding="utf-8")
        
        return filtered_rdf_data

if __name__ == "__main__":
    endpoint = SPARQLEndpoint("http://dbpedia.org/sparql")
    endpoint.connect()

    # test querying
    # result = endpoint.query("SELECT ?label WHERE { <http://dbpedia.org/resource/Asturias> rdfs:label ?label }")
    # print(result)

    # test binding
    uri = endpoint.lookup_term("asturias")
    print(uri)

    # test describe
    item = endpoint.describe_resource(uri)
    # endpoint.write_resource_to_file(uri, "asturias.ttl")

    # test related items
    # related_items = endpoint.get_related_items(uri)
    # print("Related items: ")
    # for item in related_items:
    #     print(" - " + item)

    # test filtering
    filtered_data = endpoint.filter_rdf_triples(item, "turtle", ["http://dbpedia.org/ontology/abstract"])
    with open("asturias_filtered.ttl", "wb") as file:
        file.write(filtered_data)

    endpoint.disconnect()
