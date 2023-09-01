import csv
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS
from tqdm import tqdm
from rdflib.tools.rdf2dot import rdf2dot
from subprocess import call

class MovieLensRDFConverter:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.graph = Graph()
        self.ns = Namespace("http://example.org/movielens#")
        self.genre_ns = Namespace("http://example.org/genre#")
        self.occupation_ns = Namespace("http://example.org/occupation#")

    def convert(self):
        self._convert_occupations()
        self._convert_users()
        self._convert_genres()
        self._convert_items()
        self._convert_ratings()

        return self.graph

    def _convert_users(self):
        user_file = f"{self.data_dir}/u.user"
        num_lines = sum(1 for line in open(user_file,'r'))
        with open(user_file, "r") as file:
            for line in tqdm(file, desc="Importing users", ncols=150, total=num_lines):
                if line.strip() == "":
                    continue
                user_id, age, gender, occupation, zip_code = line.strip().split("|")
                user_uri = self.ns[f"user{user_id}"]

                self.graph.add((user_uri, RDF.type, self.ns.User))
                self.graph.add((user_uri, self.ns.age, Literal(age)))
                self.graph.add((user_uri, self.ns.gender, Literal(gender)))
                self.graph.add((user_uri, self.ns.occupation, self.occupation_ns[occupation]))

    def _convert_items(self):
        item_file = f"{self.data_dir}/u.item"
        num_lines = sum(1 for line in open(item_file,'r'))
        with open(item_file, "r", encoding="ISO-8859-1") as file:
            for line in tqdm(file, desc="Importing items", ncols=150, total=num_lines):
                if line.strip() == "":
                    continue
                fields = line.strip().split("|")
                item_id = fields[0]
                movie_title = fields[1]
                release_date = fields[2]
                imdb_url = fields[4]

                item_uri = self.ns[f"item{item_id}"]
                self.graph.add((item_uri, RDF.type, self.ns.Item))
                self.graph.add((item_uri, RDFS.label, Literal(movie_title)))
                self.graph.add((item_uri, self.ns.release_date, Literal(release_date)))
                self.graph.add((item_uri, self.ns.imdb_url, Literal(imdb_url)))

                # Extract genres
                genres = fields[5:]
                for index, genre in enumerate(genres):
                    if genre == "1":
                        genre_uri = self.genre_ns[f"genre{index}"]
                        self.graph.add((item_uri, self.ns.genre, genre_uri))

    def _convert_ratings(self):
        ratings_file = f"{self.data_dir}/u.data"
        num_lines = sum(1 for line in open(ratings_file,'r'))
        with open(ratings_file, "r") as file:
            for line in tqdm(file, desc="Importing ratings", ncols=150, total=num_lines):
                if line.strip() == "":
                    continue
                user_id, item_id, rating, timestamp = line.strip().split("\t")
                rating_uri = self.ns[f"rating{user_id}_{item_id}"]

                self.graph.add((rating_uri, RDF.type, self.ns.Rating))
                self.graph.add((rating_uri, self.ns.user, self.ns[f"user{user_id}"]))
                self.graph.add((rating_uri, self.ns.item, self.ns[f"item{item_id}"]))
                self.graph.add((rating_uri, self.ns.rating, Literal(rating)))
                self.graph.add((rating_uri, self.ns.timestamp, Literal(timestamp)))

    def _convert_genres(self):
        genre_file = f"{self.data_dir}/u.genre"
        num_lines = sum(1 for line in open(genre_file,'r'))
        with open(genre_file, "r") as file:
            for line in tqdm(file, desc="Importing genres", ncols=150, total=num_lines):
                if line.strip() == "":
                    continue
                genre_name, genre_id = line.strip().split("|")
                genre_uri = self.genre_ns[f"genre{genre_id}"]

                self.graph.add((genre_uri, RDF.type, self.ns.Genre))
                self.graph.add((genre_uri, RDFS.label, Literal(genre_name)))

    def _convert_occupations(self):
        occupation_file = f"{self.data_dir}/u.occupation"
        num_lines = sum(1 for line in open(occupation_file,'r'))
        with open(occupation_file, "r") as file:
            for line in tqdm(file, desc="Importing occupations", ncols=150, total=num_lines):
                if line.strip() == "":
                    continue
                occupation = line.strip()
                occupation_uri = self.occupation_ns[occupation]

                self.graph.add((occupation_uri, RDF.type, self.ns.Occupation))

    def load_graph(self, file_name):
        print("Loading graph")
        self.graph.parse(f"{self.data_dir}/{file_name}", format="turtle", encoding="utf-8")

    def write_graph(self, file_name):
        self.graph.serialize(f"{self.data_dir}/{file_name}", format="turtle", encoding="utf-8")

    def visualize(self):
        print("Visualization: Creating dot file")
        # with open(f"{self.data_dir}/graph.dot", "w", encoding="utf-8") as dot_file:
        #     rdf2dot(self.graph, dot_file)

        print("Visualization: Creating png file")
        # Generate the visualization using GraphViz
        call(["dot", "-Tpng", f"{self.data_dir}/graph.dot", "-o", f"{self.data_dir}/graph.png"])

    def get_graph(self):
        return self.graph
    
    def rdf_to_csv(self, filename, graph=None):
        if graph is None:
            graph = self.graph
        with open(filename, 'w', newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            for s, p, o in graph:
                writer.writerow([str(s), str(p), str(o)])

if __name__ == "__main__":
    converter = MovieLensRDFConverter("./data/ml-100k")
    #rdf_graph = converter.convert()
    #rdf_graph.serialize("./data/ml-100k/graph.turtle", format="turtle")
    converter.load_graph("graph.turtle")
    converter.rdf_to_csv("./data/ml-100k/graph.csv")
    #converter.visualize()
