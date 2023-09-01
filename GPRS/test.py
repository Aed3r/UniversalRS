# Step 1: Load the Knowledge Graph
import rdflib
import numpy as np

g = rdflib.Graph()
g.parse("data/ml-100k/graph.turtle", format="turtle")

# Step 2: Install and Import Necessary Libraries
# Note: You'll need to install the libraries first. Use pip:
# !pip install rdflib pykeen

from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

# Convert RDF graph to triples
triples_list = [(str(s), str(p), str(o)) for s, p, o in g]

# Convert list to numpy array
triples_array = np.array(triples_list)

# Create the TriplesFactory
tf = TriplesFactory.from_labeled_triples(triples=triples_array)

# Step 3: Generate Embeddings
result = pipeline(
    model='TransE',
    training=tf,
    testing=tf,
    stopper='early',
    training_kwargs=dict(num_epochs=50),  # You can adjust the number of epochs
)

# The embeddings can be accessed with result.model.entity_embeddings
embeddings = result.model.entity_embeddings()

# Step 4: Save Embeddings
# Save to a file or use as needed

# For example, to save to a numpy file:
import numpy as np
np.save("data/ml-100k/embeddings.npy", embeddings)

print("Embeddings generated and saved!")
