import networkx as nx
import pandas as pd
from rdflib import Graph


class Node2VecEmbeddings:
    def __init__(self, dimensions=128, walk_length=30, num_walks=200, workers=4):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.model = None
        self.node_mapping = {}
        self.edge_df = None

    def load_graph(self, graph_file):
        print("Loading RDF graph...")
        graph = Graph()
        graph.parse(graph_file, format='rdf')
        return graph

    def encode_nodes(self, graph):
        self.node_mapping = {}
        nodes = list(graph.all_nodes())

        for i, node in enumerate(nodes):
            self.node_mapping[node] = i

        print("Nodes encoded as integers.")

    def create_edge_dataframe(self, graph):
        edge_data = []
        for s, p, o in graph:
            source = self.node_mapping.get(s)
            target = self.node_mapping.get(o)
            edge_type = p
            edge_data.append((source, target, edge_type))

        columns = ["Source", "Target", "Edge Type"]
        self.edge_df = pd.DataFrame(edge_data, columns=columns)

        print("Edge DataFrame created.")

    def generate_embeddings(self, graph_file):
        graph = self.load_graph(graph_file)

        self.encode_nodes(graph)
        self.create_edge_dataframe(graph)

        nx_graph = nx.Graph()
        for s, p, o in graph:
            source = self.node_mapping.get(s)
            target = self.node_mapping.get(o)
            edge_type = p
            nx_graph.add_edge(source, target, edge_type=edge_type)

        print("Initializing node2vec model...")
        node2vec = Node2Vec(nx_graph, dimensions=self.dimensions, walk_length=self.walk_length,
                            num_walks=self.num_walks, workers=self.workers)

        print("Generating node2vec embeddings...")
        self.model = node2vec.fit(window=10, min_count=1, batch_words=4)

        print("Embeddings generated for all nodes!")

    def add_new_nodes(self, new_nodes):
        nx_graph = self.model.graph
        nx_graph.add_nodes_from(new_nodes)

        print("Updating node2vec model with new nodes...")
        self.model = self.model.fit(window=10, min_count=1, batch_words=4)

    def get_embeddings(self, nodes):
        embeddings = {node: self.model.wv[node] for node in nodes}
        return embeddings


# Function to convert RDF graph to edge DataFrame
def rdf_graph_to_edge_dataframe(graph_file):
    graph = Graph()
    graph.parse(graph_file, format='rdf')

    edge_data = []
    for s, p, o in graph:
        edge_data.append((s, o, p))

    columns = ["Source Node", "Target Node", "Edge Type"]
    edge_df = pd.DataFrame(edge_data, columns=columns)

    return edge_df


# Example usage:
embeddings_generator = Node2VecEmbeddings(dimensions=128, walk_length=30, num_walks=200, workers=4)

# Generate embeddings and create edge DataFrame
embeddings_generator.generate_embeddings('your_graph.rdf')

# Get the encoded node mapping and edge DataFrame
node_mapping = embeddings_generator.node_mapping
edge_df = embeddings_generator.edge_df

# Add new nodes and update the embeddings
new_nodes = ['new_node1', 'new_node2', 'new_node3']
embeddings_generator.add_new_nodes(new_nodes)

# Get the updated embeddings for the new nodes
new_node_embeddings = embeddings_generator.get_embeddings(new_nodes)

# Convert RDF graph to edge DataFrame
rdf_graph_file = 'your_graph.rdf'
edge_df = rdf_graph_to_edge_dataframe(rdf_graph_file)

# Print the encoded node mapping
print("Node Mapping:")
for node, encoding in node_mapping.items():
    print(f"{node} -> {encoding}")

# Print the edge DataFrame
print("Edge DataFrame:")
print(edge_df)
