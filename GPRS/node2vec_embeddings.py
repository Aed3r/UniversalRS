import networkx as nx
from node2vec import Node2Vec
from rdflib import Graph
from tqdm import tqdm
import time

class Node2VecEmbeddings:
    def __init__(self, graph_file, dimensions=128, walk_length=30, num_walks=200, workers=4):
        self.graph_file = graph_file
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.model = None

    def load_graph(self):
        print("Loading RDF graph...")
        start_time = time.time()

        graph = Graph()
        graph.parse(self.graph_file, format='rdf')

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"RDF graph loaded in {elapsed_time:.2f} seconds")

        return graph

    def generate_embeddings(self):
        graph = self.load_graph()

        print("Creating NetworkX graph...")
        start_time = time.time()

        nx_graph = nx.Graph()
        for s, p, o in graph:
            nx_graph.add_edge(s, o)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"NetworkX graph created in {elapsed_time:.2f} seconds")

        print("Initializing node2vec model...")
        start_time = time.time()

        node2vec = Node2Vec(nx_graph, dimensions=self.dimensions, walk_length=self.walk_length,
                            num_walks=self.num_walks, workers=self.workers)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Node2Vec model initialized in {elapsed_time:.2f} seconds")

        print("Generating node2vec embeddings...")
        start_time = time.time()

        self.model = node2vec.fit(window=10, min_count=1, batch_words=4)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Node2Vec embeddings generated in {elapsed_time:.2f} seconds")

        print("Embeddings generated for all nodes!")

    def add_new_nodes(self, new_nodes):
        nx_graph = self.model.graph
        nx_graph.add_nodes_from(new_nodes)

        print("Updating node2vec model with new nodes...")
        start_time = time.time()

        self.model = self.model.fit(window=10, min_count=1, batch_words=4)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Node2Vec model updated with new nodes in {elapsed_time:.2f} seconds")

    def get_embeddings(self, nodes):
        embeddings = {}
        progress_bar = tqdm(total=len(nodes), desc="Retrieving Embeddings")

        for node in nodes:
            start_time = time.time()
            embeddings[node] = self.model.wv[node]
            end_time = time.time()
            elapsed_time = end_time - start_time
            progress_bar.set_postfix({"Elapsed Time": f"{elapsed_time:.2f} s"})
            progress_bar.update(1)

        progress_bar.close()
        print("Embeddings retrieved for all nodes!")

        return embeddings


# Example usage:
embeddings_generator = Node2VecEmbeddings('your_graph.rdf', dimensions=128, walk_length=30, num_walks=200, workers=4)

# Generate embeddings
embeddings_generator.generate_embeddings()

# New nodes to be added
new_nodes = ['new_node1', 'new_node2', 'new_node3']

# Add the new nodes and update the embeddings
embeddings_generator.add_new_nodes(new_nodes)

# Get the updated embeddings for the new nodes
new_node_embeddings = embeddings_generator.get_embeddings(new_nodes)

# Print the embeddings for the new nodes
for node, embedding in new_node_embeddings.items():
    print(f"Embedding for '{node}': {embedding}")
