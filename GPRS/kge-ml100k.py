import pathlib
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

HERE = pathlib.Path(__file__).resolve().parent
PATH = HERE.parent.joinpath("data", "ml-100k", "graph.csv")

tf = TriplesFactory.from_path(PATH)

training, testing = tf.split()

result = pipeline(
    training=training,
    testing=testing,
    model='TransE',
    epochs=5,  # short epochs for testing - you should go higher
)

result.save_to_directory(HERE.parent.joinpath("data", "ml-100k", "kge-results"))