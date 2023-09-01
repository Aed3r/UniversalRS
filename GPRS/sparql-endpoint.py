from SPARQLWrapper import SPARQLWrapper, JSON

class SPARQLEndpoint:
    def __init__(self, endpoint_url):
        self.endpoint_url = endpoint_url
        self.sparql = None

    def connect(self):
        """
        Initializes a connection to the SPARQL endpoint.
        """
        self.sparql = SPARQLWrapper(self.endpoint_url)

    def disconnect(self):
        """
        Disconnects from the SPARQL endpoint (for this example, it's simply setting the connection to None).
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

if __name__ == "__main__":
    endpoint = SPARQLEndpoint("http://dbpedia.org/sparql")
    endpoint.connect()
    result = endpoint.query("SELECT ?label WHERE { <http://dbpedia.org/resource/Asturias> rdfs:label ?label }")
    print(result)
    endpoint.disconnect()
