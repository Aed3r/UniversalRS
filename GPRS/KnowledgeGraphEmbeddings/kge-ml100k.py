import pathlib
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import rdflib
import numpy as np
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS

NS = Namespace("http://example.org/movielens#")
HERE = pathlib.Path(__file__).resolve().parent
PATH = HERE.parent.joinpath("data", "ml-100k", "graph.turtle")
FILTER = {str(NS.Rating), str(NS.User)}

g = rdflib.Graph()
g.parse(PATH, format="turtle")

# Convert RDF graph to triples
triples_list = [(str(s), str(p), str(o)) for s, p, o in g]

# Filter triples
nodes_to_filter = {triple[0] for triple in triples_list if triple[1] == str(RDF.type) and triple[2] in FILTER}

filtered_triples = [triple for triple in triples_list if triple[0] not in nodes_to_filter and triple[2] not in nodes_to_filter]

# Convert list to numpy array
triples_array = np.array(filtered_triples)

# Create the TriplesFactory
tf = TriplesFactory.from_labeled_triples(triples=triples_array)
#tf = TriplesFactory.from_path(PATH)

training, testing = tf.split()

result = pipeline(
    training=training,
    testing=testing,
    model='TransE',
    epochs=5,  # short epochs for testing - you should go higher
    use_tqdm=True,
)

result.save_to_directory(HERE.parent.joinpath("data", "ml-100k", "kge-results-content-based"))