import signal
from SPARQLWrapper import SPARQLWrapper, JSON, TURTLE, SPARQLExceptions
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from matplotlib import patheffects
import numpy as np
import requests
import json
from rdflib import Graph, URIRef
from tqdm import tqdm
import pandas as pd
from http.client import RemoteDisconnected
import time
from collections import defaultdict
import subprocess
import os
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import traceback
from itertools import combinations
from queue import PriorityQueue
import itertools

HOME = "/nethome/1060546/"
DATASETS = {
    'Movielens1M': {
        'mapping': HOME + 'data/LODrecsys-datasets/Movielens1M/MappingMovielens2DBpedia-1.2.tsv',
        'ratings': HOME + 'data/ml-1m/ratings.dat',
        'ratingsDelimiter': '::',
        'ratingsHasHeader': False
    },
    'Librarything': {
        'mapping': HOME + 'data/LODrecsys-datasets/LibraryThing/MappingLibrarything2DBpedia-1.2.tsv',
        'ratings': HOME + 'data/LibraryThing/reviews.tsv',
        'ratingsDelimiter': '::',
        'ratingsHasHeader': False
    },
    'Lastfm': {
        'mapping': HOME + 'data/LODrecsys-datasets/LastFM/MappingLastfm2DBpedia-1.2.tsv',
        'ratings': HOME + 'data/lastFM/user_artists.dat',
        'ratingsDelimiter': '\t',
        'ratingsHasHeader': True
    }
}
BLAZEGRAPH_NAMESPACE = "namespace/kb/sparql"
BLAZEGRAPH_PORT = 19999
BLAZEGRAPH_LOCATION = HOME + "blazegraph/"

class SPARQLEndpoint:
    def __init__(self, endpoint_url):
        self.endpoint_url = endpoint_url
        self.sparql = None
        self.lookup_url = "https://lookup.dbpedia.org/api/search"
        self.blazegraphRunning = False
        self.dbpediaLock = threading.Lock()
        self.blazegraphLock = threading.Lock()

    def connectDBpedia(self):
        """
        Initializes connection
        """
        self.sparql = SPARQLWrapper(self.endpoint_url)

    def disconnectDBpedia(self):
        """
        Disconnects from endpoint
        """
        self.sparql = None

    def launch_blazegraph(self, url=None, verbose=False, port=BLAZEGRAPH_PORT):
        if url is not None:
            self.blazegraphURL = url
        else:
            command = ["java", "-server", "-Xmx70g", "-Djetty.port="+str(port), "-jar", "blazegraph.jar"]
            self.blazegraphProcess = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, cwd=BLAZEGRAPH_LOCATION)

            print("Waiting for Blazegraph to start...")

            # Loop to read each line of output
            for line in iter(self.blazegraphProcess.stdout.readline, ''):
                if verbose:
                    print(line, end='')

                # Check if the desired message is in the output
                if line.startswith("Go to http"):
                    self.blazegraphURL = line.split(" ")[2]
                    break
                elif line.startswith("ERROR"):
                    print("An error occurred while starting Blazegraph: " + line)
                    print("Blazegraph may already be running. Here are the processes using blazegraph.jnl:")
                    os.chdir(HOME + "blazegraph/")
                    os.system("fuser -v blazegraph.jnl")
                    exit()
            
        # Test the connection
        try:
            print("Testing Blazegraph connection...")
            sparql = SPARQLWrapper(self.blazegraphURL + BLAZEGRAPH_NAMESPACE)
            sparql.setQuery("""
                SELECT ?p ?o WHERE {
                    <http://dbpedia.org/resource/Toy_Story> ?p ?o .
                }
                LIMIT 1
            """)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()

            if not results["results"]["bindings"] or len(results["results"]["bindings"]) == 0:
                print("Blazegraph data is not properly loaded.")
                exit()
        except Exception as e:
            print(f"Blazegraph is not accessible. Error: {e}")
            exit()
        
        self.blazegraphRunning = True
        self.blazegraph = SPARQLWrapper(self.blazegraphURL + BLAZEGRAPH_NAMESPACE)

        print("Blazegraph is accessible at " + self.blazegraphURL + BLAZEGRAPH_NAMESPACE)

    def stop_blazegraph(self):
        if self.blazegraphRunning:
            print("Stopping Blazegraph...")
            os.killpg(os.getpgid(self.blazegraphProcess.pid), signal.SIGTERM)
            self.blazegraphRunning = False

    def query(self, query_string, useBlazegraph=False, verbose = False):
        """
        Executes a query against the SPARQL endpoint and returns the results in JSON format.
        """
        if not self.sparql:
            raise Exception("Not connected to any SPARQL endpoint.")
        
        # If Blazegraph is not running, blazegraph is running but useBlazegraph is not set, or the DBpedia lock is available, use DBpedia
        if (not useBlazegraph or not self.blazegraphRunning) and self.dbpediaLock.acquire(blocking=(not self.blazegraphRunning)):
            if verbose:
                print("Using DBpedia...", flush=True)
            try:
                self.sparql.setQuery(query_string)
                self.sparql.setReturnFormat(JSON)
                return self.sparql.query().convert()
            except RemoteDisconnected:
                print("Remote disconnected. Reconnecting...")
                time.sleep(5)
                self.connectDBpedia()
                return None
            except SPARQLExceptions.QueryBadFormed as e:
                if "timeout" in e.args[0].lower():
                    if self.blazegraphRunning:
                        print("Timeout error. Trying Blazegraph...")
                        return self.query(query_string, useBlazegraph=True)
                    else:
                        print(f"Timeout error: {e.args[0]}")
                        return None
                print(f"Bad query: {e.args[0]}")
                return None
            except Exception as e:
                print(f"Error querying the SPARQL endpoint: {e}")
                return None
            finally:
                self.dbpediaLock.release()
        # Otherwise, use Blazegraph
        elif self.blazegraphRunning:
            with self.blazegraphLock:
                if verbose:
                    print("Using Blazegraph...", flush=True)
                self.blazegraph.setQuery(query_string)
                self.blazegraph.setReturnFormat(JSON)
                return self.blazegraph.query().convert()
        
    
    def lookup_term(self, term, n=5, field="query", exact=False, typeName=None):
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
            "maxResults": n,
            "format": "JSON_RAW"
        }

        if exact:
            params[field + "Exact"] = "true"

        if typeName:
            params["typeName"] = typeName

        response = requests.get(self.lookup_url, params=params)

        if response.status_code == 200:
            data = response.json()
            if data['docs'] and len(data['docs']) > 0:
                return data['docs']
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
        self.sparql.setReturnFormat(JSON)

        try:
            results = self.sparql.query().convert()
            return results
        except Exception as e:
            print(f"Error querying the SPARQL endpoint: {e}")
            return None
        
    def get_type_of_uri(self, uri):
        query_string = f"""
        SELECT ?mostSpecificType WHERE {{
        <{uri}> rdf:type ?type .
        
        FILTER(strstarts(str(?type), str(dbo:)))

        OPTIONAL {{
            ?moreSpecificType rdfs:subClassOf+ ?type .
            <{uri}> rdf:type ?moreSpecificType .
            FILTER(strstarts(str(?moreSpecificType), str(dbo:)))
        }}

        BIND(coalesce(?moreSpecificType, ?type) AS ?mostSpecificType)
        }}
        GROUP BY ?mostSpecificType
        """
        results = self.query(query_string)
        try:
            return results["results"]["bindings"][0]["mostSpecificType"]["value"]
        except RemoteDisconnected:
            time.sleep(5)
            self.connectDBpedia()
            return None
        except:
            return None
    
    def get_dbpedia_type(self, uri):
        try:
            headers = {'Range': 'bytes=0-12000'}

            # Fetch the web page
            url = uri.replace("resource", "page")
            response = requests.get(url, headers=headers)
            page_content = response.content

            # Parse the HTML using BeautifulSoup
            soup = BeautifulSoup(page_content, 'html.parser')

            # Find the target span element
            span_elements = soup.find_all('span', {'class': 'text-nowrap'})

            for span in span_elements:
                if "An Entity of Type:" in span.text:
                    link_element = span.find('a')
                    if link_element:
                        link = link_element.get('href')
                        return link
                        break
        except Exception as e:
            print(f"Error querying the type of '{uri}': {e}")
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
    
    def get_item_attributes(self, uri, attributes):
        """
        Efficiently retrieves the values of the specified attributes of an item from the SPARQL endpoint.

        Args:
            uri (str): The URI of the item to retrieve the attributes for.
            attributes (list): The list of attributes to retrieve.

        Returns:
            dict: A dictionary mapping attribute names to their values.
        """
        attributes = [(attr, attr.split('/')[-1]) for attr in attributes]
        optional_clauses = "\n".join([f"  OPTIONAL {{ <{uri}> <{attr}> ?{attr_name} . }}" for attr, attr_name in attributes])
        query_string = f"""
        SELECT *
        WHERE {{
            {optional_clauses}
        }}
        """
        query_string = query_string.replace('\n', ' ').strip()
        results = self.query(query_string)

        if results == None or len(results["results"]["bindings"]) == 0 or len(results["results"]["bindings"][0]) == 0:
            print(f"Failed to retrieve data for {uri}.")
            return {}
        
        entity_data = defaultdict(set)
        for binding in results["results"]["bindings"]:
            for _, attr_name in attributes:
                if attr_name in binding:
                    value = binding[attr_name]["value"]
                    if value != "":
                        entity_data[attr_name].add(binding[attr_name]["value"])

        # Convert set to dictionary of lists
        entity_data = {k: list(v) for k, v in entity_data.items()}
        
        return entity_data
    
    def get_similar_items(self, rdf_type, entity_uri, attributes, entity_data, limit):
        """
        Query n similar items of a specified type and attributes from the SPARQL endpoint.

        Args:
            rdf_type (str): The type of items to query for.
            entity_uri (str): The URI of the item the attributes come from (to exclude from the results).
            attributes (dict): The dictionary of attribute-value pairs for filtering.
            entity_data (dict): The dictionary of lists containing attribute-value pairs for filtering.
            limit (int): The maximum number of items to retrieve.

        Returns:
            list: A list containing the URIs of the retrieved items.
        """
        attributes = [(attr, attr.split('/')[-1]) for attr in attributes]
        attr_names = [attr_name for _, attr_name in attributes]

        #attribute_string = " ".join(f"?{attr_names}" for _, attr_names in attributes)

        filter_clauses = []
        for attr_name, values in entity_data.items():
            if attr_name not in attr_names:
                continue
            if len(values) > 1:
                value_list = []
                for value in values:
                    if value.startswith("http"):
                        value = value.replace("\\", "")
                        value_list.append(f"<{value}>")
                    else:
                        value = value.replace('"', '')
                        value_list.append(f'"{value}"')
                value_list = ",".join(value_list)
                filter_clauses.append(f"?{attr_name} IN ({value_list})")
            else:
                value = values[0]
                if value.startswith("http"):
                    value = value.replace("\\", "")
                    filter_clauses.append(f"?{attr_name} = <{value}>")
                else:
                    value = value.replace('"', '')
                    filter_clauses.append(f'?{attr_name} = "{value}"')
        filter_clauses.append(f"?item != <{entity_uri}>")
        filter_string = " && ".join(filter_clauses)
        
        triple_string = " ".join(f'?item <{attr}> ?{attr_names} .' for attr, attr_names in attributes)
        
        query_string = f"""
        SELECT DISTINCT ?item
        WHERE {{
            ?item a <{rdf_type}> .
            {triple_string}
            FILTER({filter_string})
        }}
        LIMIT {limit}
        """
        query_string = query_string.replace('\n', ' ').strip()
        
        results = self.query(query_string)

        if results == None:
            return []

        items = []
        for result in results["results"]["bindings"]:
            items.append(result["item"]["value"])
        
        return items
    
    def get_similar_items_iter_binary_search(self, rdf_type, entity_uri, attributes, n):
        """
        Retrieves n items of a specified type and attributes from the SPARQL endpoint.
        If n items are not found, the method iteratively removes the last attribute in the dictionary
        and tries again until at least n items are found.

        Args:
            rdf_type (str): The type of items to query for.
            entity_uri (str): The URI of the item the attributes come from (to exclude from the results).
            attributes (list): The list of attributes used to match candidates, sorted by priority.
            n (int): The number of items to retrieve.

        Returns:
            entity_uri (str): The URI of the entity the items were retrieved for.
            best_candidates (list): A list containing the top n URIs of the retrieved items.
            log (dict): A dictionary containing the number of queries necessary, the time taken and score of each candidate.
        """
        print(f"Finding similar items for {entity_uri}...")

        start = time.time()

        # Get entity data for the given attributes
        entity_data = endpoint.get_item_attributes(entity_uri, attributes)
        if entity_data == None or len(entity_data) == 0:
            print(f"Failed to retrieve data for {entity_uri}.", flush=True)
            return entity_uri, "failed (no initial data found)", "failed (no initial data found)"

        tot = len(attributes)
        low = 0
        high = tot
        result_uris = {}
        most_specific_subset_i = tot
        queriesNecessary = 0
        while True:
            mid = (low + high) // 2

            if mid in result_uris: 
                print("Should not happen!", flush=True)
                print("results: " + str(result_uris) + " mid: " + str(mid) + " low: " + str(low) + " high: " + str(high), flush=True)
                most_specific_subset_i = tot
                break

            # Query the SPARQL endpoint
            result = self.get_similar_items(rdf_type, entity_uri, attributes[:mid], entity_data, n)
            queriesNecessary += 1

            result_uris[mid] = result

            if len(result) == 0:
                # If no results were found, try again with a smaller subset of attributes
                if mid-1 in result_uris and len(result_uris[mid-1]) > 0:
                    most_specific_subset_i = mid-1
                    break

                if mid == 1:
                    most_specific_subset_i = 0
                    break

                high = mid
                continue
            else:
                # If we have enough items, try again with a larger subset of attributes
                if mid+1 in result_uris and len(result_uris[mid+1]) == 0:
                    most_specific_subset_i = mid
                    break

                if mid == tot:
                    most_specific_subset_i = tot
                    break

                low = mid
                continue
        
        best_candidates = []
        logging = []
        i = most_specific_subset_i
        while i > 0 and len(best_candidates) < n:
            if i in result_uris:
                for uri in result_uris[i]:
                    if uri not in best_candidates:
                        best_candidates.append(uri)
                        logging.append((uri, i))
            else:
                result = self.get_similar_items(rdf_type, entity_uri, attributes[:i], entity_data, n)
                queriesNecessary += 1
                for uri in result:
                    if uri not in best_candidates:
                        best_candidates.append(uri)
                        logging.append((uri, i))
            i -= 1 

        end = time.time()

        print(f"Found {max(len(best_candidates), n)} similar items for {entity_uri}.", flush=True)

        # If we still don't have enough items, just return what we have
        return entity_uri, best_candidates[:n], {"totalQueries": queriesNecessary, "timeTaken": round(end-start, 2), "candidates": logging[:n]}

    def get_similar_items_iter_divide_conquer(self, rdf_type, entity_uri, ranked_attributes, n):
        """
        Retrieves n items of a specified type and attributes from the SPARQL endpoint using the divide and conquer approach.

        Args:
            rdf_type (str): The type of items to query for.
            entity_uri (str): The URI of the item the attributes come from (to exclude from the results).
            ranked_attributes (list): The list of attributes used to match candidates, sorted by priority.
            n (int): The number of items to retrieve.

        Returns:
            entity_uri (str): The URI of the entity the items were retrieved for.
            best_candidates (list): A list containing the top n URIs of the retrieved items.
            log (dict): A dictionary containing the number of queries necessary, the time taken and score of each candidate.
        """
        print(f"Finding similar items for {entity_uri}...")

        start = time.time()

        entity_data = endpoint.get_item_attributes(entity_uri, ranked_attributes)
        if entity_data == None or len(entity_data) == 0:
            print(f"Failed to retrieve data for {entity_uri}.", flush=True)
            return entity_uri, "failed (no initial data found)", "failed (no initial data found)"

        all_items = {}
        tot_searches = 0
        for attr_combo in combinations(ranked_attributes, 2):
            tot_searches += 1
            items = endpoint.get_similar_items(rdf_type, entity_uri, attr_combo, entity_data, n)
            for item in items:
                if item not in all_items:
                    all_items[item] = set()
                all_items[item].update(attr_combo)
        
        # Divide and conquer
        def divide_and_conquer(attributes):
            nonlocal tot_searches, all_items

            if len(attributes) <= 2:
                return -1

            mid = len(attributes) // 2
            left, right = attributes[:mid], attributes[mid:]
            
            found = divide_and_conquer(left)
            found += divide_and_conquer(right)

            if found == 0:
                return 0
            
            # Update items
            tot_searches += 1
            items = endpoint.get_similar_items(rdf_type, entity_uri, attributes, entity_data, n)
            for item in items:
                if item not in all_items:
                    all_items[item] = set()
                all_items[item].update(attributes)

            return len(items)
        
        divide_and_conquer(ranked_attributes)

        sorted_items = {k: v for k, v in sorted(all_items.items(), key=lambda item: len(item[1]), reverse=True)}

        candidates = list(sorted_items.keys())[:n]
        log = {}
        log["timeTaken"] = round(time.time() - start, 2)
        log["totalQueries"] = tot_searches
        log["candidates"] = {}
        for candidate in candidates:
            log["candidates"][candidate] = list(sorted_items[candidate])

        print(f"Found {len(candidates)} similar items for {entity_uri}.", flush=True)
        
        # Return top n items
        return entity_uri, candidates, log
    
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
    
    def load_instance_types(self, filename):
        """
        Loads the instance types from a file into a dictionary.

        Args:
            filename (str): The name of the file containing the instance types.

        Returns:
            dict: A dictionary mapping URIs to their types.
        """
        print(f"Loading instance types from {filename}...")
        # instance_types = {}
        # with open(filename, 'r') as file:
        #     for line in tqdm(file, desc="Loading instance types", unit="lines", total=6895561):
        #         uri, _, rdf_type, _ = line.strip().split(' ')
        #         instance_types[uri] = rdf_type
        instance_types = pd.read_csv(filename, sep=' ', header=None, names=['uri', 'a', 'rdf_type', 'dot'])
        print(f"Loaded {len(instance_types)} instance types.")
        return instance_types
    
    def get_redirect(self, uri):
        """
        Tests whether a URI redirects to another URI.

        Args:
            uri (str): The URI to test.

        Returns:
            str: The URI to which the given URI redirects, or None if no redirect found.
        """
        query = f"""
            SELECT ?redirect
            WHERE {{
                <{uri}> <http://dbpedia.org/ontology/wikiPageRedirects> ?redirect .
            }}
            """
        results = self.query(query)

        bindings = results['results']['bindings']
        
        if len(bindings) > 0:
            return bindings[0]['redirect']['value']
        else:
            return None
        
    def check_type(self, uri, types):
        query_string = f"""
        SELECT ?type WHERE {{
            <{uri}> a ?type .
            FILTER(strstarts(str(?type), str(dbo:)))
        }}
        """
        results = self.query(query_string)

        if results == None:
            return None

        try:
            for binding in results["results"]["bindings"]:
                if binding["type"]["value"] in types:
                    return True
            return False
        except:
            return None

def test_querying(endpoint):
    result = endpoint.query("SELECT ?label WHERE { <http://dbpedia.org/resource/Asturias> rdfs:label ?label }")
    print(result)

def test_binding(endpoint):
    docs = endpoint.lookup_term("batman", typeName="film")
    if docs:
        for res in docs:
            print(res['label'][0], res['resource'][0])
    else:
        print("No results found.")

def test_describe(endpoint):
    uri = "http://dbpedia.org/resource/Asturias"
    item = endpoint.describe_resource(uri)
    endpoint.write_resource_to_file(uri, "asturias.ttl")

def test_type_retrieval(endpoint):
    dataset="Lastfm"
    mapping = pd.read_csv(DATASETS[dataset]['mapping'], sep='\t', names=['ItemID', 'Title', 'DBpediaURI'])
    types = {}
    dbpediaTypes = {}
    for i, row in mapping.iterrows():
        uri = row["DBpediaURI"]
        type = endpoint.get_type_of_uri(uri)
        dbpediaType = endpoint.get_dbpedia_type(uri)

        if not type:
            type = "failed"

        if not dbpediaType:
            dbpediaType = "failed"
        
        if type in types:
            types[type] += [uri]
        else:
            types[type] = [uri]

        if dbpediaType in dbpediaTypes:
            dbpediaTypes[dbpediaType] += [uri]
        else:
            dbpediaTypes[dbpediaType] = [uri]
        
        mapping.at[i, "Type"] = type
        mapping.at[i, "DBpediaType"] = dbpediaType
        print(f"{i}: {uri} - {type} -- {dbpediaType}")
    
    # lastChange = -1
    # n = 0
    # while lastChange == -1 or lastChange > 0:
    #     lastChange = 0
    #     n += 1
    #     for i, row in tqdm(mapping.iterrows(), desc="Repairing mappings (n="+str(n)+")"):
    #         if row["Type"] == "failed":
    #             uri = row["DBpediaURI"]
    #             type = endpoint.get_type_of_uri(uri)
    #             if type:
    #                 mapping.at[i, "Type"] = type
    #                 lastChange += 1
    #         if row["DBpediaType"] == "failed":
    #             uri = row["DBpediaURI"]
    #             dbpediaType = endpoint.get_dbpedia_type(uri)
    #             if dbpediaType:
    #                 mapping.at[i, "DBpediaType"] = dbpediaType
    #                 lastChange += 1 

    print("Types: ")
    for type in types:
        print(f"{type}: {len(types[type])}")
    print("DBpedia types: ")
    for dbpediaType in dbpediaTypes:
        print(f"{dbpediaType}: {len(dbpediaTypes[dbpediaType])}")
    #save results
    with open("data/results/typeRetrieval/types_"+dataset+".json", 'w+') as file:
        json.dump(types, file)
    with open("data/results/typeRetrieval/dbpediaTypes_"+dataset+".json", 'w+') as file:
        json.dump(dbpediaTypes, file)
    mapping.to_csv("data/results/typeRetrieval/mapping_"+dataset+".csv", sep='\t', index=False)


def test_related_items(endpoint):
    uri = "http://dbpedia.org/resource/Asturias"
    related_items = endpoint.get_related_items(uri)
    print("Related items: ")
    for item in related_items:
        print(" - " + item)

def test_filtering(endpoint):
    uri = "http://dbpedia.org/resource/Asturias"
    filtered_data = endpoint.filter_rdf_triples(uri, "turtle", ["http://dbpedia.org/ontology/abstract"])
    with open("asturias_filtered.ttl", "wb") as file:
        file.write(filtered_data)

def test_similar_items(endpoint):
    entity = "http://dbpedia.org/resource/Batman_(1989_film)"
    type = "http://dbpedia.org/ontology/Film"
    attributes = ["http://dbpedia.org/ontology/starring", 
                  "http://dbpedia.org/ontology/director", 
                  "http://dbpedia.org/ontology/writer",
                  "http://dbpedia.org/ontology/producer",
                  "http://dbpedia.org/ontology/musicComposer",
                  "http://dbpedia.org/property/music",
                  "http://dbpedia.org/ontology/distributor",
                  "http://dbpedia.org/ontology/language",
                  "http://dbpedia.org/ontology/cinematography",
                  "http://dbpedia.org/ontology/country",
                  "http://dbpedia.org/ontology/editing",
                  "http://dbpedia.org/property/studio",
                  "http://dbpedia.org/property/extra",
                  "http://dbpedia.org/property/screenplay",
                  "http://dbpedia.org/property/genre"]
    n = 15
    dataset = "Movielens1M"
    alg = "divConq"
    save_loc_candidates = "data/results/candidateSel/candidates_"+dataset+"_"+alg+".json"
    save_loc_log = "data/results/candidateSel/candidates_log_"+dataset+"_"+alg+".json"
    if alg == "binSearch":
        alg = endpoint.get_similar_items_iter_binary_search
    elif alg == "divConq":
        alg = endpoint.get_similar_items_iter_divide_conquer

    # entity = "http://dbpedia.org/resource/Kingpin"
    # entity_data = endpoint.get_item_attributes(entity, attributes)
    # similar_items = endpoint.get_similar_items(type, entity, attributes, entity_data, 10)
    # print("Similar items: ")
    # for item in similar_items:
    #     print(" - " + item)

    candidates = {}
    full_log = {}
    mapping = pd.read_csv(DATASETS[dataset]['mapping'], sep='\t', names=['ItemID', 'Title', 'DBpediaURI'])

    if os.path.exists(save_loc_candidates):
        with open(save_loc_candidates, 'r') as file:
            candidates = json.load(file)
        print(f"Loaded {len(candidates)} candidates.")

        if len(candidates) > 0:
            success = []
            for entity, similar_items in candidates.items():
                if isinstance(similar_items, list):
                    success.append(entity)
            print(f"Found {len(success)} successful candidates.")

            mapping = mapping[~mapping['DBpediaURI'].isin(success)]
            
    if os.path.exists(save_loc_log):
        with open(save_loc_log, 'r') as file:
            full_log = json.load(file)
        print(f"Loaded {len(full_log)} log entries.")

    endpoint.launch_blazegraph()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for i, row in mapping.iterrows():
            try:
                future = executor.submit(alg, type, row["DBpediaURI"], attributes, n)
                futures.append(future)
            except Exception as e:
                print(f"An error occurred while processing candidates: {e}")
                traceback.print_exc()
        for i, future in tqdm(enumerate(futures), total=len(futures), desc="Processing candidates"):
            try:
                result = future.result(timeout=900)  # 15 minutes * 60 seconds = 900 seconds

                if result:
                    entity, similar_items, logging = result
                    candidates[entity] = similar_items
                    full_log[entity] = logging

                    if i % 20 == 0:
                        with open(save_loc_candidates, 'w+') as file:
                            json.dump(candidates, file)
                        with open(save_loc_log, 'w+') as file:
                            json.dump(full_log, file)
                        print(f"Saved results for {i} entities.")
            except TimeoutError:
                print("A candidate selection took too long to process and was terminated.")
            except Exception as e:
                print(f"An error occurred while selecting a candidate: {e}")
                traceback.print_exc()

            exception = future.exception()
            if exception is not None:
                print(f"Exception in thread: {exception}")
    
    with open(save_loc_candidates, 'w+') as file:
        json.dump(candidates, file)
    with open(save_loc_log, 'w+') as file:
        json.dump(full_log, file)

def analyze_candidate_selection(dataset="Movielens1M", algName="binSearch"): # binSearch or divConq
    num_successful_searches = 0
    num_failed_searches = 0
    total_queries_necessary = 0
    total_time_taken = 0.0
    total_candidates = 0
    score_frequency = defaultdict(int)
    attribute_specificity = [] # how many attributes on average are matched by the retrieved items. 
    accuracy = 0.0
    total_found = 0

    print(f"Analyzing candidate selection results for {dataset} using {algName}...")

    save_loc_log = "data/results/candidateSel/candidates_log_"+dataset+"_"+algName+".json"
    mapping = pd.read_csv(DATASETS[dataset]['mapping'], sep='\t', names=['ItemID', 'Title', 'DBpediaURI'])

    with open(save_loc_log, 'r') as file:
        full_log = json.load(file)

    # Loop through the data to calculate statistics
    for key, value in full_log.items():
        if isinstance(value, dict):  # Successful search
            if len(value['candidates']) == 0:
                num_failed_searches += 1
                continue

            num_successful_searches += 1
            total_queries_necessary += value['totalQueries']
            total_time_taken += value['timeTaken']
            total_candidates += len(value['candidates'])


            if isinstance(value["candidates"], list):
                acc = 0
                for item, score in value["candidates"]:
                    score_frequency[score] += 1
                    attribute_specificity.append(score)
                    if item in mapping["DBpediaURI"].values:
                        total_found += 1
                        acc += 1
                if len(value["candidates"]) > 0:
                    accuracy += acc / len(value["candidates"])
            elif isinstance(value["candidates"], dict):
                acc = 0
                for item in value["candidates"].keys():
                    score = len(value["candidates"][item])
                    score_frequency[score] += 1
                    attribute_specificity.append(score)
                    if item in mapping["DBpediaURI"].values:
                        total_found += 1
                        acc += 1
                if len(value["candidates"]) > 0:
                    accuracy += acc / len(value["candidates"])
        else:  # Failed search
            num_failed_searches += 1

    # Calculate averages for successful searches
    if num_successful_searches > 0:
        avg_queries_necessary = total_queries_necessary / num_successful_searches
        avg_time_taken = total_time_taken / num_successful_searches
        avg_candidates = total_candidates / num_successful_searches
        sorted_scores = sorted(score_frequency.items(), key=lambda x: x[0])
        attribute_specificity = sum(attribute_specificity) / len(attribute_specificity)
        accuracy = accuracy / num_successful_searches
    else:
        avg_queries_necessary = 0
        avg_time_taken = 0
        avg_candidates = 0
        sorted_scores = []
        attribute_specificity = 0

    # Compile the statistics
    statistics = {
        'Number of Successful Searches': num_successful_searches,
        'Number of Failed Searches': num_failed_searches,
        'Average Number of Queries Necessary': avg_queries_necessary,
        'Average Time Taken (seconds)': avg_time_taken,
        'Average Number of Candidate Items': avg_candidates,
        'Score Frequency': sorted_scores,
        'Attribute Specificity': attribute_specificity,
        'Accuracy wrt existing mapping': accuracy,
        'Total found from existing mapping': total_found
    }

    # Score distribution pie chart
    labels = [str(score) for score, _ in sorted_scores]
    sizes = [freq for _, freq in sorted_scores]
    colors = ['#EAD1DC', '#C6A4A4', '#D9C8AE', '#A2A392', '#B3D3C1', '#D0E0EB']

    # Merge the last scores into one
    labels[len(colors)-1] = str(labels[len(colors)-1]) + "+"
    for i in range(len(colors), len(labels)):
        sizes[len(colors)-1] += sizes[i]
    labels = labels[:len(colors)]
    sizes = sizes[:len(colors)]

    percentages = [(count / sum(sizes) * 100) for count in sizes]
    labels = [f"{label} ({percent:.1f}%)" for label, percent in zip(labels, percentages)]

    plt.figure(figsize=(10, 10))
    plt.pie(sizes, startangle=140, colors=colors, textprops={'fontsize': 12, 'color': 'black'}, labels=labels, autopct='%1.1f%%')
    #plt.legend(labels, title="Scores", loc="best")
    #plt.title('Distribution of Candidate Item Scores')

    plt.savefig("data/results/candidateSel/scoreDistribution_"+dataset+"_"+algName+".png")

    for key, value in statistics.items():
        print(f"{key}: {value}")

def test_redirect(endpoint):
    dataset = "Movielens1M"
    mapping = pd.read_csv(DATASETS[dataset]['mapping'], sep='\t', names=['ItemID', 'Title', 'DBpediaURI'])
    for i, row in tqdm(mapping.iterrows(), total=len(mapping), desc="Searching redirects"):
        label = row["Title"]
        uri = endpoint.lookup_term(label)
        redirect = endpoint.get_redirect(uri)
        if redirect:
            print(f"Redirect found: {uri} -> {redirect}")

def test_type_check(endpoint):
    uri = "http://dbpedia.org/resource/Toy_Story"
    types = ["http://dbpedia.org/ontology/Film"]
    print(endpoint.check_type(uri, types))

if __name__ == "__main__":
    endpoint = SPARQLEndpoint("http://dbpedia.org/sparql")
    endpoint.connectDBpedia()

    # test_querying(endpoint)
    # test_binding(endpoint)
    # test_describe(endpoint)
    # test_type_retrieval(endpoint)
    # test_related_items(endpoint)
    # test_filtering(endpoint)
    #test_similar_items(endpoint)
    analyze_candidate_selection(algName="divConq")
    # test_redirect(endpoint)
    # test_type_check(endpoint)

    endpoint.disconnectDBpedia()
    endpoint.stop_blazegraph()