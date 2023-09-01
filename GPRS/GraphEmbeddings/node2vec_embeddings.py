import networkx as nx
import pandas as pd
from node2vec import Node2Vec
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS
from tqdm import tqdm
from rdflib.tools.rdf2dot import rdf2dot
from subprocess import call

class Node2VecEmbeddings:
    def __init__(self, dimensions=128, walk_length=30, num_walks=200, workers=4):
        self.graph = None
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.model = None
        self.node_mapping = {}
        self.edge_df = None
        self.nodes_df = None
        self.ns = None

    def load_turtle_graph(self, graph_file, namespace):
        print("Loading RDF graph...")
        self.ns = Namespace(namespace)
        self.graph = Graph()
        self.graph.parse(graph_file, format='turtle')
        return self.graph

    def encode_graph(self, filterNodeTypes=None):
        """Encode the RDF graph as a node and edge DataFrame.
        
        Args:
            filterNodeTypes (list): List of node types to filter out. If None, all nodes are included.
        """

        print("Encoding RDF graph...")
        if self.graph is None:
            raise Exception("No graph loaded. Load a graph first using load_turtle_graph()")
        edge_data = []
        nodes_data = []
        self.node_mapping = {}
        node_count = 0
        for s, p, o in self.graph:
            # Encode nodes as integers
            for node in [s, o]:
                if node not in self.node_mapping:
                    # Find node type
                    node_type = self.graph.value(node, RDF.type)
                    if type is not None and node_type in filterNodeTypes:
                        break # one of the nodes is not of the specified type, we skip this triple
                    # Update mapping
                    self.node_mapping[node] = node_count
                    node_count += 1
                    # Add node to DataFrame
                    nodes_data.append((node, node_type))
            else:
                # Add edge to DataFrame
                source = self.node_mapping.get(s)
                target = self.node_mapping.get(o)
                edge_type = p
                edge_data.append((source, target, edge_type))

        self.edge_df = pd.DataFrame(edge_data, columns=["Source", "Target", "Type"])
        self.nodes_df = pd.DataFrame(nodes_data, columns=["Node", "Type"])

        print("Node and Edge DataFrames created.")

    def generate_embeddings(self):
        self.get_nodes_df(self.graph)
        self.create_edge_dataframe(self.graph)

        nx_graph = nx.Graph()
        for s, p, o in self.graph:
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


if __name__ == '__main__':
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
