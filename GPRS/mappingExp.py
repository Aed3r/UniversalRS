from SPARQLWrapper import SPARQLWrapper, JSON, TURTLE
from bs4 import BeautifulSoup
import pandas as pd
import requests
from tqdm import tqdm
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

HOME = "/nethome/1060546/"
DATASETS = {
    "Movielens1M": HOME + 'data/LODrecsys-datasets/Movielens1M/MappingMovielens2DBpedia-1.2.tsv',
    "Librarything": HOME + 'data/LODrecsys-datasets/LibraryThing/MappingLibrarything2DBpedia-1.2.tsv',
    "Lastfm": HOME + 'data/LODrecsys-datasets/LastFM/MappingLastfm2DBpedia-1.2.tsv',
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

        try:
            return self.sparql.query().convert()
        except Exception as e:
            print(f"Error querying the SPARQL endpoint: {e}")
            return None

    def lookup_term(
        self, term, n=5, field="query", exact=False, typeFilter=None
    ):
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
            "format": "JSON_RAW",
            "exactMatchBoost": 100,
            "prefixMatchBoost": 1,
        }

        if exact:
            params[field + "Exact"] = "true"

        response = requests.get(self.lookup_url, params=params)

        if response.status_code == 200:
            data = response.json()
            if data["docs"] and len(data["docs"]) > 0:
                docs = data["docs"]
            
                # Filter by type
                if typeFilter:
                    toRemove = []
                    for item in docs:
                        if not "resource" in item:
                            continue
                        if not self.check_type(item["resource"][0], typeFilter):
                            toRemove += [item]
                    for item in toRemove:
                        docs.remove(item)
                return docs
        return None

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

        if results == None:
            return None

        bindings = results["results"]["bindings"]

        if len(bindings) > 0:
            return bindings[0]["redirect"]["value"]
        else:
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

        if results == None:
            return None

        try:
            return results["results"]["bindings"][0]["mostSpecificType"]["value"]
        except:
            return None
    
    def get_dbpedia_type(self, uri):
        try:
            headers = {'Range': 'bytes=0-12000'}

            # Fetch the web page
            url = "https://dbpedia.org/page/Asturias"
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
                        print(f"Found the link: {link}")
                        break
        except Exception as e:
            print(f"Error querying the type of '{uri}': {e}")
            return None
        
    def check_type(self, uri, types):
        query_string = f"""
        SELECT ?type WHERE {{
            <{uri}> a ?type .
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


def clean_movie_title(title):
    # Extract the movie name and remove everything starting from the first occurrence of '('
    movie_name = title.split("(", 1)[0].strip()

    # Check if ", The" is present in the movie name
    if ", The" in movie_name:
        # Reorder the title with "The" at the beginning
        cleaned_title = f'The {movie_name.replace(", The", "")}'.strip()
    elif ", A" in movie_name:
        cleaned_title = f'A {movie_name.replace(", A", "")}'.strip()
    else:
        # If no definite article, keep the title as it is
        cleaned_title = movie_name

    return cleaned_title


def load_mapping(dataset):
    if dataset not in DATASETS.keys():
        print("Invalid dataset specified.")
        exit()

    print(f"Loading {dataset} mapping...")
    mapping = pd.read_csv(
        DATASETS[dataset], sep="\t", names=["ItemID", "Title", "ManualDBpediaURI"]
    )
    mapping["Title"] = mapping["Title"].apply(clean_movie_title)
    print(f"Loaded {len(mapping)} item mappings.")

    return mapping


def analyze(mapping, correct, incorrect, missing, exportTo):
    correct_mappings = len(correct)
    incorrect_mappings = len(incorrect)
    missing_mappings = len(missing)

    colors = ['#F79F79', '#80CED7', '#FFD700']

    labels = ['Correct mappings', 'Incorrect mappings', 'Missing mappings']
    sizes = [correct_mappings, incorrect_mappings, missing_mappings]

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 12, 'color': 'black'})

    plt.title('Mapping Results', fontsize=14, fontweight='bold')

    plt.savefig(exportTo + '_results.png')



    duplicate_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    for index, row in mapping.iterrows():
        if str(row["AutoDBpediaURI"]) == "nan":
            print("missing:", row["Title"],
                  row["ManualDBpediaURI"], row["AutoDBpediaURI"])
        elif row["ManualDBpediaURI"] != row["AutoDBpediaURI"]:
            # print( "wrong", row["Title"], row["ManualDBpediaURI"], "->", row["AutoDBpediaURI"], row["DuplicateCount"])
            duplicate_count = row["DuplicateCount"]
            if duplicate_count in duplicate_counts:
                duplicate_counts[duplicate_count] += 1

    # Plotting the pie chart
    colors = ['#EAD1DC', '#C6A4A4', '#D9C8AE', '#A2A392', '#B3D3C1', '#D0E0EB']

    labels = [f'{count} ambiguous results' for count in duplicate_counts.keys()]
    sizes = list(duplicate_counts.values())

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors,textprops={'fontsize': 12, 'color': 'black'})

    #plt.title('Distribution of Ambiguous Results for Incorrect Mappings', fontsize=14, fontweight='bold')
    plt.savefig(exportTo + "_duplicates.png")


if __name__ == "__main__":
    print("Mapping experiment on Lastfm with type filter")
    endpoint = SPARQLEndpoint("http://dbpedia.org/sparql")
    endpoint.connect()

    # Load the mapping
    dataset = "Lastfm"
    mapping = load_mapping(dataset)  # Movielens1M, Librarything, Lastfm
    typeFilter = ["http://dbpedia.org/ontology/Artist", "http://dbpedia.org/ontology/MusicalArtist", "http://umbel.org/umbel/rc/MusicalPerformer", "http://dbpedia.org/ontology/Band", "http://umbel.org/umbel/rc/Artist"] # None or a DBpedia type URI (eg http://dbpedia.org/ontology/Film)
    filterName = "artist"

    # mapping = mapping.iloc[:int(len(mapping)/20)]

    # Lookup all terms in the mapping and get the most likely DBpedia URI.
    redirected = []
    unfoundTypeAvg = []
    for index, row in tqdm(
        mapping.iterrows(), total=len(mapping), desc="Creating mappings"
    ):
        docs = endpoint.lookup_term(
            row["Title"], field="label", exact=True, typeFilter=typeFilter, n=10
        )
        # if any(char in ',.!?;:-_()[]{}<>|\\/`~@#$%^&*+=\'"' for char in row['Title']):
        #   title_without_punctuation = ''.join(char for char in row['Title'] if char.isalnum() or char.isspace())
        #   # Update 'Title' with the punctuation removed
        #   row['Title'] = title_without_punctuation

        if not docs or len(docs) == 0:
            docs = endpoint.lookup_term(
                row["Title"], field="query", exact=True, typeFilter=typeFilter, n=10
            )

        if not docs or len(docs) == 0:
            docs = endpoint.lookup_term(
                row["Title"], field="query", exact=False, typeFilter=typeFilter, n=10
            )

        if (not docs or len(docs) == 0) and typeFilter:
            docs = endpoint.lookup_term(
                row["Title"] + " " + filterName, field="query", exact=False, typeFilter=typeFilter, n=10
            )

        if not docs or len(docs) == 0:
            continue

        closest_match = None
        closest_match_length = float("inf")
        found = False
        exactMatchFound = False
        foundURI = None
        for item in docs:
            if not "label" in item or not "resource" in item:
                if "resource" in item:
                    print(f"{item['resource']} has no label")
                continue
            uri = item["resource"][0]
            label = item["label"][0]
            formatted_title = row["Title"].replace(" ", "_")
            formatted_label = label.replace(" ", "_")
            adjustedTitle = row["Title"].lower().replace("the", "").strip()
            adjustedLabel = label.lower().replace("the", "").strip()
            # if len(label) == len(row['Title']):
            #   mapping.at[index, 'AutoDBpediaURI'] = uri
            #   print(uri)

            # else:
            #   print('cannot find the map')

            if (adjustedLabel == adjustedTitle):
                found = True
                foundURI = uri
                break
            # elif '(film)' in label:
            #   # Pattern '(XXXX film)' found in label
            #   film_title = label.split(' (')[0]
            #   film_uri = 'http://dbpedia.org/resource/' + film_title.replace(' ', '_')
            #   if film_uri in docs:
            #       # Film URI found in the list
            #       mapping.at[index, 'AutoDBpediaURI'] = uri
            #   else:
            #       # Film URI not found in the list
            #       print(f"Cannot find URI for film '{film_title}' in this URI list")
            # elif '_(' in formatted_label and formatted_label.endswith('_film)'):
            #   mapping.at[index, 'AutoDBpediaURI'] = uri
            #   print('this uri with _film found:', uri)
            #   break

            diff = abs(len(label) - len(row['Title'])) # length difference between label and title
            common = sum(
                (
                    Counter(label.lower().split())
                    & Counter(row["Title"].lower().split())
                ).values()
            )
            if adjustedTitle in label.lower():  # a result contains the searched title
                if not exactMatchFound:  # First
                    closest_match = None
                exactMatchFound = True
                if len(row["Title"].split()) > 1: # title contains more than one word
                    if not closest_match or closest_match_length < common:
                        closest_match = uri
                        closest_match_length = common
                else: # title contains only one word
                    if not closest_match or closest_match_length > diff:
                        closest_match = uri
                        closest_match_length = common
            elif not exactMatchFound:
                if len(row["Title"].split()) > 1: # title contains more than one word
                    if not closest_match or closest_match_length < common:
                        closest_match_length = common
                        closest_match = uri
                else: # title contains only one word
                    if not closest_match or closest_match_length > diff:
                        closest_match_length = common
                        closest_match = uri
        if not found:
            foundURI = closest_match

        duplicateCount = 0
        for item in docs:
            if not "label" in item or not "resource" in item:
                continue
            label = item["label"][0]
            adjustedTitle = row["Title"].replace("The", "").lower().strip()

            if adjustedTitle in label.lower():
                duplicateCount += 1
        mapping.at[index, "DuplicateCount"] = duplicateCount

        if not foundURI:
            continue

        # Test redirect
        redirect = endpoint.get_redirect(foundURI)
        if redirect:
            mapping.at[index, "AutoDBpediaURI"] = redirect
            redirected += [(foundURI, redirect)]
        else:
            mapping.at[index, "AutoDBpediaURI"] = foundURI

    if typeFilter:
        exportTo = "/nethome/1060546/data/results/mapping_type/" + dataset
    else:
        exportTo = "/nethome/1060546/data/results/mapping_simple/" + dataset

    # Save the mapping results to a new CSV file
    mapping.to_csv(exportTo + ".csv", index=False)
    print("Mapping results saved to MappingResults.csv")

    # Save the redirected results to a CSV file
    redirected_df = pd.DataFrame(redirected, columns=['FoundURI', 'RedirectedURI'])
    redirected_df.to_csv(exportTo + "_redir.csv", index=False)
    print("Redirected results saved to RedirectedResults.csv")

    # Compare the manual and automatic mappings. Count correct mappings, incorrect mappings, and missing mappings.
    correct = []
    incorrect = []
    missing = []
    for index, row in tqdm(
        mapping.iterrows(), total=len(mapping), desc="Comparing mappings"
    ):
        if row["ManualDBpediaURI"] == row["AutoDBpediaURI"]:
            correct += [row]
        elif str(row["AutoDBpediaURI"]) == "nan":
            missing += [row]
        else:
            incorrect += [row]

    print(f"\n\n{len(correct)} correct mappings:")
    for row in correct:
        print(f" - {row['Title']} -> {row['AutoDBpediaURI']}")
    print(f"\n\n{len(incorrect)} incorrect mappings:")
    for row in incorrect:
        print(f" - {row['Title']} -> {row['AutoDBpediaURI']} not {row['ManualDBpediaURI']}")
    print(f"\n\n{len(missing)} missing mappings:")
    for row in missing:
        print(f" - {row['Title']} -> _ not {row['ManualDBpediaURI']}")

    print(f"Accuracy: {round(len(correct) * 100 / len(mapping), 2)}%")

    print(f"{len(redirected)} redirects found:")
    for found, redir in redirected:
        print(f" - {found} -> {redir}")

    analyze(mapping, correct, incorrect, missing, exportTo)

    # Disconnect from the endpoint.
    endpoint.disconnect()
