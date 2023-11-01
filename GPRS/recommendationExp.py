from itertools import islice
import json
import os
import subprocess
import time
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sparql_endpoint import SPARQLEndpoint
import threading
import math
import multiprocessing
from SPARQLWrapper import SPARQLWrapper, JSON
from pyrdf2vec.walkers import *
from pyrdf2vec.samplers import *
from pyrdf2vec.graphs import KG
from pyrdf2vec.embedders import Embedder
from cachetools import LRUCache
from pyrdf2vec import RDF2VecTransformer
import traceback
import datetime
import lz4.frame
from gensim.models.word2vec import Word2Vec as W2V
import attr
from pyrdf2vec.typings import Embeddings, Entities, SWalk
import signal

HOME = "/nethome/1060546/"
DBPEDIA_URL = "https://dbpedia.org/sparql"
BLAZEGRAPH_NAMESPACE = "namespace/kb/sparql"
BLAZEGRAPH_PORT = 19999
BLAZEGRAPH_LOCATION = HOME + "blazegraph/"
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
EMBEDDING_STORE_LOC = HOME + 'data/results/embeddings/'
FAIL_LOC = HOME + 'data/results/embeddings/fails.json'
SIMILARITY_LOC = HOME + 'data/results/sim/'
RECOMMENDATION_SCORES_LOC = HOME + 'data/results/scores/'
FIGURES_LOC = HOME + 'data/results/figures/'
EVAL_FILE_LOC = HOME + 'data/results/eval.json'
CONFIGS_LOC = HOME + 'data/results/configs/'
WALKS_LOC = HOME + 'data/results/walks/'
W2V_MODEL_LOC = HOME + 'data/results/w2v/'
SKIP_VERIFY = True # Set to True to skip verification of the items in the KG
CACHE_SIZE = 10000
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

class DotDict(dict):     
    """dot.notation access to dictionary attributes"""      
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val
    __setattr__ = dict.__setitem__     
    __delattr__ = dict.__delitem__

class SentenceIterator(object):
    def __init__(self, configName, item_ids):
        self.path = WALKS_LOC + configName + '/'
        self.ids = item_ids
        self.runCount = 0
    
    def __iter__(self):
        self.runCount += 1
        return self._generator()

    def _generator(self):
        for item_id in tqdm(self.ids, desc="Loading walks (run " + str(self.runCount) + ")"):
            path = self.path + str(item_id) + '.lz4'
            if not os.path.exists(path):
                print(f"File '{path}' not found.")
                continue
            try:
                with lz4.frame.open(path, "r") as f:
                    walks = json.loads(f.read().decode('utf-8'))
                    for walk in walks:
                        yield walk
            except Exception as e:
                print(f"An error occurred while loading walks from file '{str(item_id)}.lz4': {e}")
                #traceback.print_exc()

@attr.s(init=False)
class Word2Vec(Embedder):
    """Defines the Word2Vec embedding technique. Own version of pyrdf2vec's Word2Vec class that supports building the corpus iteratively and saving the model.

    SEE: https://radimrehurek.com/gensim/models/word2vec.html

    Attributes:
        _model: The gensim.models.word2vec model.
            Defaults to None.
        kwargs: The keyword arguments dictionary.
            Defaults to { min_count=0 }.

    """

    kwargs = attr.ib(init=False, default=None)
    _model = attr.ib(init=False, type=W2V, default=None, repr=False)

    def __init__(self, **kwargs):
        self.kwargs = {
            "min_count": 0,
            **kwargs,
        }
        self._model = W2V(**self.kwargs)

    def fit(
        self, walksIterator, configName, is_update: bool = False
    ) -> Embedder:
        """Fits the Word2Vec model based on provided walks.

        Args:
            walks: The walks to create the corpus to to fit the model.
            is_update: True if the new walks should be added to old model's
                walks, False otherwise.
                Defaults to False.

        Returns:
            The fitted Word2Vec model.

        """
        print("Building W2V vocabulary...")
        self._model.build_vocab(walksIterator, update=is_update)
        print("Training W2V model...")
        self._model.train(
            walksIterator,
            total_examples=self._model.corpus_count,
            epochs=self._model.epochs,
        )
        print("W2V model trained. Saving...")
        self._model.save(W2V_MODEL_LOC + configName + '.bin')
        
        return self

    def load(self, configName: str) -> Embedder:
        """Loads the Word2Vec model from the provided path.

        Args:
            configName: The name of the config file used to create the model.

        Returns:
            The loaded Word2Vec model.

        """
        self._model = W2V.load(W2V_MODEL_LOC + configName + '.bin')
        return self

    def transform(self, entities: Entities) -> Embeddings:
        """The features vector of the provided entities.

            Args:
                entities: The entities including test entities to create the
                embeddings. Since RDF2Vec is unsupervised, there is no label
                leakage.

        Returns:
            The features vector of the provided entities.

        """
        if not all([entity in self._model.wv for entity in entities]):
            raise ValueError(
                "The entities must have been provided to fit() first "
                "before they can be transformed into a numerical vector."
            )
        return [self._model.wv.get_vector(entity) for entity in entities]


class GPRS:
    def __init__(self, config):
        config = DotDict(config)
        self.config = config
        self.mapping = None
        #self.sparql = SPARQLEndpoint(ENDPOINT_URL)
        #self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_HOST, db=0)
        self.ratings = None
        self.similarity_df = None
        self.train_ratings = None
        self.test_ratings = None
        self.recommendations = None
        self.embedding_store = {}
        self.lock = threading.Lock()
        self.fileLock = threading.Lock()
        self.save_counter = 0
        self.fails = []
        self.dbpediaLock = threading.Lock()
        self.dbpediaKG = None
        self.blazegraphKG = None
        self.results = None
        self.cache = LRUCache(maxsize=CACHE_SIZE)
        self.cacheLock = threading.Lock()
        self.blazegraphRunning = False
        
    def return_rdf2vec_config(self, useBlazegraph): # useBlazegraph = True means use Blazegraph, False means use DBpedia
        # Sampler config
        if self.config.params.walkerConfig.sampler == "ObjFreqSampler":
            sampler = ObjFreqSampler(
                split=self.config.params.walkerConfig.samplerConfig.split, 
                inverse=self.config.params.walkerConfig.samplerConfig.inverse)
        elif self.config.params.walkerConfig.sampler == "ObjPredFreqSampler":
            sampler = ObjPredFreqSampler(
                split=self.config.params.walkerConfig.samplerConfig.split, 
                inverse=self.config.params.walkerConfig.samplerConfig.inverse)
        elif self.config.params.walkerConfig.sampler == "PageRankSampler":
            sampler = PageRankSampler(
                split=self.config.params.walkerConfig.samplerConfig.split, 
                inverse=self.config.params.walkerConfig.samplerConfig.inverse, 
                alpha=self.config.params.walkerConfig.samplerConfig.alpha)
        elif self.config.params.walkerConfig.sampler == "PredFreqSampler":
            sampler = PredFreqSampler(
                split=self.config.params.walkerConfig.samplerConfig.split, 
                inverse=self.config.params.walkerConfig.samplerConfig.inverse)
        elif self.config.params.walkerConfig.sampler == "UniformSampler":
            sampler = UniformSampler()
        elif self.config.params.walkerConfig.sampler == "WideSampler":
            sampler = WideSampler(
                split=self.config.params.walkerConfig.samplerConfig.split, 
                inverse=self.config.params.walkerConfig.samplerConfig.inverse)
        else:
            print("Invalid sampler specified.")
            exit()

        # Walker self.config
        bgNJobs = 10
        if self.config.params.walker == "AnonymousWalker":
            if not useBlazegraph:
                walkers = [
                    AnonymousWalker(
                        max_depth=self.config.params.walkerConfig.max_depth, 
                        max_walks=self.config.params.walkerConfig.max_walks, 
                        with_reverse=self.config.params.walkerConfig.with_reverse, 
                        sampler=sampler)
                ]
            else:
                walkers = [
                    AnonymousWalker(
                        max_depth=self.config.params.walkerConfig.max_depth, 
                        max_walks=self.config.params.walkerConfig.max_walks, 
                        with_reverse=self.config.params.walkerConfig.with_reverse, 
                        sampler=sampler,
                        n_jobs=bgNJobs)
                ]
        elif self.config.params.walker == "CommunityWalker":
            if not useBlazegraph:
                walkers = [
                    CommunityWalker(
                        max_depth=self.config.params.walkerConfig.max_depth, 
                        max_walks=self.config.params.walkerConfig.max_walks, 
                        with_reverse=self.config.params.walkerConfig.with_reverse, 
                        sampler=sampler)
                ]
            else:
                walkers = [
                    CommunityWalker(
                        max_depth=self.config.params.walkerConfig.max_depth, 
                        max_walks=self.config.params.walkerConfig.max_walks, 
                        with_reverse=self.config.params.walkerConfig.with_reverse, 
                        sampler=sampler,
                        n_jobs=bgNJobs)
                ]
        elif self.config.params.walker == "HALKWalker":
            if not useBlazegraph:
                walkers = [
                    HALKWalker(
                        max_depth=self.config.params.walkerConfig.max_depth,
                        max_walks=self.config.params.walkerConfig.max_walks, 
                        with_reverse=self.config.params.walkerConfig.with_reverse, 
                        sampler=sampler)
                ]
            else:
                walkers = [
                    HALKWalker(
                        max_depth=self.config.params.walkerConfig.max_depth,
                        max_walks=self.config.params.walkerConfig.max_walks, 
                        with_reverse=self.config.params.walkerConfig.with_reverse, 
                        sampler=sampler,
                        n_jobs=bgNJobs)
                ]
        elif self.config.params.walker == "NGramWalker":
            if not useBlazegraph:
                walkers = [
                    NGramWalker(
                        max_depth=self.config.params.walkerConfig.max_depth,
                        max_walks=self.config.params.walkerConfig.max_walks, 
                        with_reverse=self.config.params.walkerConfig.with_reverse, 
                        sampler=sampler)
                ]
            else:
                walkers = [
                    NGramWalker(
                        max_depth=self.config.params.walkerConfig.max_depth,
                        max_walks=self.config.params.walkerConfig.max_walks, 
                        with_reverse=self.config.params.walkerConfig.with_reverse, 
                        sampler=sampler,
                        n_jobs=bgNJobs)
                ]
        elif self.config.params.walker == "RandomWalker":
            if not useBlazegraph:
                walkers = [
                    RandomWalker(
                        max_depth=self.config.params.walkerConfig.max_depth,
                        max_walks=self.config.params.walkerConfig.max_walks, 
                        with_reverse=self.config.params.walkerConfig.with_reverse, 
                        sampler=sampler)
                ]
            else:
                walkers = [
                    RandomWalker(
                        max_depth=self.config.params.walkerConfig.max_depth,
                        max_walks=self.config.params.walkerConfig.max_walks, 
                        with_reverse=self.config.params.walkerConfig.with_reverse, 
                        sampler=sampler,
                        n_jobs=bgNJobs)
                ]
        elif self.config.params.walker == "SplitWalker":
            if not useBlazegraph:
                walkers = [
                    SplitWalker(
                        max_depth=self.config.params.walkerConfig.max_depth,
                        max_walks=self.config.params.walkerConfig.max_walks, 
                        with_reverse=self.config.params.walkerConfig.with_reverse, 
                        sampler=sampler)
                ]
            else:
                walkers = [
                    SplitWalker(
                        max_depth=self.config.params.walkerConfig.max_depth,
                        max_walks=self.config.params.walkerConfig.max_walks, 
                        with_reverse=self.config.params.walkerConfig.with_reverse, 
                        sampler=sampler,
                        n_jobs=bgNJobs)
                ]
        elif self.config.params.walker == "WalkletWalker":
            if not useBlazegraph:
                walkers = [
                    WalkletWalker(
                        max_depth=self.config.params.walkerConfig.max_depth,
                        max_walks=self.config.params.walkerConfig.max_walks, 
                        with_reverse=self.config.params.walkerConfig.with_reverse, 
                        sampler=sampler)
                ]
            else:
                walkers = [
                    WalkletWalker(
                        max_depth=self.config.params.walkerConfig.max_depth,
                        max_walks=self.config.params.walkerConfig.max_walks, 
                        with_reverse=self.config.params.walkerConfig.with_reverse, 
                        sampler=sampler,
                        n_jobs=bgNJobs)
                ]
        elif self.config.params.walker == "WLWalker":
            if not useBlazegraph:
                walkers = [
                    WLWalker(
                        max_depth=self.config.params.walkerConfig.max_depth,
                        max_walks=self.config.params.walkerConfig.max_walks, 
                        with_reverse=self.config.params.walkerConfig.with_reverse, 
                        sampler=sampler)
                    ]
            else:
                walkers = [
                    WLWalker(
                        max_depth=self.config.params.walkerConfig.max_depth,
                        max_walks=self.config.params.walkerConfig.max_walks, 
                        with_reverse=self.config.params.walkerConfig.with_reverse, 
                        sampler=sampler,
                    n_jobs=bgNJobs)
                ]
        else:
            print("Invalid walker specified.")
            exit()

        # Embedder self.config
        if self.config.params.embedder == "FastText":
            print("FastText is not supported")
            #embedder = FastText()
        elif self.config.params.embedder == "Word2Vec":
            embedder = Word2Vec(
                epochs=self.config.params.embedderConfig.epochs, 
                sg=self.config.params.embedderConfig.sg, 
                workers=15, 
                vector_size=self.config.params.embedderConfig.vector_size,
                window=self.config.params.embedderConfig.window,
                negative=self.config.params.embedderConfig.negative)

        if self.config.params.lit_weight == 0:
            literals = []
        else:
            literals = self.config.params.literals
        if not useBlazegraph:
            kg = KG(
                DBPEDIA_URL,
                skip_predicates=self.config.params.skip_predicates,
                literals=literals,
                mul_req=True,
                cache=self.cache,
                cacheLock=self.cacheLock,
                skip_verify=SKIP_VERIFY
            )
        else:
            kg = KG(
                self.blazegraphURL + BLAZEGRAPH_NAMESPACE,
                skip_predicates=self.config.params.skip_predicates,
                literals=literals,
                mul_req=True,
                cache=self.cache,
                cacheLock=self.cacheLock,
                skip_verify=SKIP_VERIFY
            )

        return embedder, walkers, kg

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

        print("Blazegraph is accessible at " + self.blazegraphURL + BLAZEGRAPH_NAMESPACE)

    def stop_blazegraph(self):
        if self.blazegraphRunning:
            print("Stopping Blazegraph...")
            os.killpg(os.getpgid(self.blazegraphProcess.pid), signal.SIGTERM)
            self.blazegraphRunning = False

    def load_mapping(self):
        dataset = self.config.dataset
        if dataset not in DATASETS.keys():
            print("Invalid dataset specified.")
            exit()
        
        print(f"Loading {dataset} mapping...")
        self.mapping = pd.read_csv(DATASETS[dataset]['mapping'], sep='\t', names=['ItemID', 'Title', 'DBpediaURI'])
        print(f"Loaded {len(self.mapping)} item mappings.")

    def load_ratings(self):
        dataset = self.config.dataset
        if dataset not in DATASETS.keys():
            print("Invalid dataset specified.")
            exit()
        
        print(f"Loading {dataset} ratings...")
        try:
            # Read the file into a DataFrame
            self.ratings = pd.read_csv(DATASETS[dataset]['ratings'], delimiter=DATASETS[dataset]['ratingsDelimiter'], header=None, engine='python', index_col=False)

            if DATASETS[dataset]['ratingsHasHeader']:
                self.ratings = self.ratings.iloc[1:]
                # Convert fields
                self.ratings[0] = self.ratings[0].astype(int)
                self.ratings[1] = self.ratings[1].astype(int)
                self.ratings[2] = self.ratings[2].astype(float)

                if len(self.ratings.columns) == 4:
                    self.ratings[3] = self.ratings[3].astype(int)

            # Rename the columns based on the expected format
            if len(self.ratings.columns) == 4: 
                self.ratings.columns = ['UserID', 'ItemID', 'Rating', 'Timestamp']
            elif len(self.ratings.columns) == 3:
                self.ratings.columns = ['UserID', 'ItemID', 'Rating']

        except FileNotFoundError:
            print(f"File {DATASETS[dataset]['ratings']} not found.")

        except Exception as e:
            print(f"An error occurred while loading the file: {e}")

        print(f"Loaded {len(self.ratings)} ratings.")

    def data_preprocessing(self):
        # Remove duplicate ratings (keep the one with the latest timestamp)
        print("Removing duplicate ratings...")
        tot = len(self.ratings)
        if 'Timestamp' not in self.ratings.columns:
            self.ratings = self.ratings.drop_duplicates(['UserID', 'ItemID'])
        else:
            self.ratings = self.ratings.sort_values('Timestamp', ascending=False).drop_duplicates(['UserID', 'ItemID']).sort_index()
        if tot - len(self.ratings) != 0:
            print(f"Removed {tot - len(self.ratings)} duplicate ratings.")
        else:
            print("No duplicate ratings found.")

        # Remove items that are not in the mapping
        print("Removing items that are not in the mapping...")
        tot = len(self.ratings)
        self.ratings = self.ratings[self.ratings['ItemID'].isin(self.mapping['ItemID'])]
        if tot - len(self.ratings) != 0:
            print(f"Removed {tot - len(self.ratings)} ratings ({round((tot - len(self.ratings)) * 100 / tot, 2)}%.)")
        else:
            print("No ratings that are not in the mapping found.")

        # Remove the top 1% most popular items 
        print("Removing top 1% most popular items...")
        popular_items = self.ratings.groupby('ItemID')['Rating'].count().reset_index().sort_values('Rating', ascending=False)
        popular_items = popular_items[popular_items['Rating'] >= popular_items['Rating'].quantile(0.99)]
        
        # Retain users with at least 50 ratings
        if self.config.dataset == 'Movielens1M':
            print("Removing users with less than 50 ratings...")
            active_users = self.ratings.groupby('UserID')['Rating'].count().reset_index().sort_values('Rating', ascending=False)
            active_users = active_users[active_users['Rating'] >= 50]

            self.ratings = self.ratings[self.ratings['UserID'].isin(active_users['UserID'])]

        # For librarything and last.fm: Remove users with less than 5 ratings and items with less than 5 ratings
        if self.config.dataset != 'Movielens1M':
            print("Removing users with less than 5 ratings...")
            active_users = self.ratings.groupby('UserID')['Rating'].count().reset_index().sort_values('Rating', ascending=False)
            active_users = active_users[active_users['Rating'] >= 5]

            print("Removing items with less than 5 ratings...")
            active_items = self.ratings.groupby('ItemID')['Rating'].count().reset_index().sort_values('Rating', ascending=False)
            active_items = active_items[active_items['Rating'] >= 5]

            self.ratings = self.ratings[self.ratings['ItemID'].isin(active_items['ItemID'])]
            self.ratings = self.ratings[self.ratings['UserID'].isin(active_users['UserID'])]

        self.ratings = self.ratings[~self.ratings['ItemID'].isin(popular_items['ItemID'])]

        # Adjust item mapping
        unique_items = self.ratings['ItemID'].unique()
        self.mapping = self.mapping[self.mapping['ItemID'].isin(unique_items)]

        if len(self.ratings) == 0:
            print("No ratings left after preprocessing.")
            exit()

        # Calculate sparsity of the ratings dataframe
        sparsity = 1 - (len(self.ratings) / (len(self.ratings['UserID'].unique()) * len(self.ratings['ItemID'].unique())))

        # Remove the timestamp column
        if 'Timestamp' in self.ratings.columns:
            self.ratings.drop('Timestamp', axis=1, inplace=True)

        # Splitting into train and test sets
        self.train_ratings, self.test_ratings = train_test_split(self.ratings, test_size=0.2, random_state=42)

        print("Dataset stats:")
        print(f"Number of users: {len(self.ratings['UserID'].unique())}")
        print(f"Number of items: {len(self.ratings['ItemID'].unique())}")
        print(f"Number of ratings: {len(self.ratings)}")
        print(f"Data sparsity: {sparsity:.2%}")
        
    def batch_extract_walks(self, item_ids, dbpedia_uris, verbose=False, useBlazegraphOnly=False):
        if verbose:
            verbosity = 2 # pyRDF2vec verbosity
        else:
            verbosity = 0

        usingDbpedia = False
        dt = datetime.datetime.now()
        if (not useBlazegraphOnly and self.dbpediaLock.acquire(False)):
            if verbose:
                print(dt.strftime("[%H:%M] ") + "Using DBpedia to process " + str(len(item_ids)) + " items.")
            usingDbpedia = True
        elif verbose:
            print(dt.strftime("[%H:%M] ") + "Using Blazegraph to process " + str(len(item_ids)) + " items.")

        embedder, walkers, kg = self.return_rdf2vec_config(useBlazegraph=not usingDbpedia)
        transformer = RDF2VecTransformer(
            embedder=embedder,
            walkers=walkers,
            verbose=verbosity
        )

        # Get walks
        try:
            #embeddings, literals = transformer.fit_transform(kg, dbpedia_uris)
            walks = transformer.get_walks(kg, dbpedia_uris)
        except Exception as e:
            print(f"An error occurred while extracting walks: {e}")
        finally:
            if usingDbpedia:
                self.dbpediaLock.release()

        del embedder
        del walkers
        del kg
        del transformer

        if walks is None or len(walks) == 0:
            print(f"Could not create walks for item {item_id}.")
            return

        #self.save_to_redis(item_id, embeddings[0], literals[0])
        for index, item_id in enumerate(item_ids):
            if item_id is None or item_id == '' or walks[index] is None or len(walks[index]) == 0:
                print(f"Could not create walks for item {item_id}.")
                continue
            
            #self.save_embeddings_to_store(item_id, embeddings[index], literals[index]) # thread safe
            self.save_walks(item_id, walks[index]) # thread safe

        del walks

        # Increment the counter and check if it's time to save
        # if saveToFile:
        #     with self.lock:
        #         self.save_counter += len(item_ids)
        #         if self.save_counter % saveEvery == 0:
        #             self.save_embeddings_to_file()
        #             print(f"Saved embeddings for {len(self.embedding_store)} items.")

        if verbose:
            print(f"Extracted and saved walks for: {str(item_ids)}")

    def extract_walks(self, verbose=True, n_jobs=3, useBlazegraphOnly=False, batchSize=10, timeout=900, bgURL=None): # n_jobs = None means use all available cores
        mapping = self.get_missing_walks()
        lastUpdate = len(mapping)
        while len(mapping) > 0:
            if not self.blazegraphRunning:
                self.launch_blazegraph(verbose=True, url=bgURL)

            print(f"Extracting walks for {len(mapping)} items...")

            total_rows = len(mapping)
            total_batches = len(mapping) // batchSize + (1 if len(mapping) % batchSize != 0 else 0)
            useBlazegraphOnly = True if len(mapping) < 2 * batchSize else False
            if len(mapping) < batchSize:
                n_jobs = 1
                useBlazegraphOnly = True
            
            with ThreadPoolExecutor(max_workers=n_jobs) as executor, tqdm(total=total_batches) as pbar:
                futures = []
                for i in range(0, total_rows, batchSize):
                    batch = mapping.iloc[i:i + batchSize]
                    batch_item_ids = batch['ItemID'].tolist()
                    batch_dbpedia_uris = batch['DBpediaURI'].tolist()

                    try:
                        future = executor.submit(self.batch_extract_walks, batch_item_ids, batch_dbpedia_uris, verbose=verbose, useBlazegraphOnly=useBlazegraphOnly)
                        future.add_done_callback(lambda p: pbar.update(1))  # Update progress bar upon task completion
                        futures.append(future)
                    except Exception as e:
                        print(f"An error occurred while processing a walk extraction batch: {e}")
                        traceback.print_exc()
                for future in futures:
                    try:
                        future.result(timeout=timeout)  # 15 minutes * 60 seconds = 900 seconds
                    except TimeoutError:
                        print("A walk extraction batch took too long to process and was terminated.")
                    except Exception as e:
                        print(f"An error occurred while processing a walk extraction batch: {e}")
                        traceback.print_exc()

                    exception = future.exception()
                    if exception is not None:
                        print(f"Exception in thread: {exception}")

            mapping = recommender.get_missing_walks(adjustMapping=0) # Remove missing items from mapping

            if lastUpdate == len(mapping):
                print("No more items can be processed.")
                recommender.get_missing_walks(adjustMapping=1) # Remove missing items from mapping
                break
        
        if self.blazegraphRunning:
            self.stop_blazegraph()

    def save_walks(self, entity_id, walks):
        if not os.path.exists(WALKS_LOC):
            os.makedirs(WALKS_LOC)
        
        if not os.path.exists(WALKS_LOC + self.config.name):
            os.makedirs(WALKS_LOC + self.config.name)

        walks = json.dumps(walks).encode('utf-8')
        compressed = lz4.frame.compress(walks)

        try:
            with open(WALKS_LOC + self.config.name + "/" + str(entity_id) + ".lz4", "wb+") as f:
                f.write(compressed)
        except Exception as e:
            print(f"An error occurred while saving walks to file: {e}")
            traceback.print_exc()

    def load_walks(self, entity_id):
        # test if file exists
        path = WALKS_LOC + self.config.name + "/" + str(entity_id) + ".lz4"
        if not os.path.isfile(path):
            print(f"Walks file '{self.config.name}/{str(entity_id)}.lz4' does not exist.")
            return None

        with lz4.frame.open(path, "r") as f:
            return json.loads(f.read().decode('utf-8'))
        
    def compress_walks(self):
         for item_id in tqdm(self.mapping['ItemID'], desc="Compressing walks"):
            try:
                path = WALKS_LOC + self.config.name + "/" + str(item_id)
                # if not os.path.isfile(path + ".json"):
                #     continue
                # with open(path + ".json", "rb") as f:
                #     walks = f.read()

                if not os.path.isfile(path + ".lz4"):
                    continue
                with open(path + ".lz4", "rb") as f:
                    walks = lz4.frame.decompress(f.read())

                with lz4.frame.open(path + ".lz4", "wb") as f:
                    f.write(walks)
                #os.remove(path + ".json")
            except Exception as e:
                print(f"An error occurred while compressing walks for item {str(item_id)}: {e}")
                traceback.print_exc()

    # adjustMapping = 0 -> return missing
    # adjustMapping = 1 -> remove missing from mapping and return missing
    # adjustMapping = 2 -> replace mapping with missing and return missing
    def get_missing_walks(self, adjustMapping=0):
        missing = []
        for item_id in tqdm(self.mapping['ItemID'], desc="Checking for missing walks"):
            if not os.path.isfile(WALKS_LOC + self.config.name + "/" + str(item_id) + ".lz4"):
                missing += [item_id]
        
        missing_df = self.mapping[self.mapping['ItemID'].isin(missing)]
        count = len(self.mapping)

        if adjustMapping == 1:
            self.mapping = self.mapping[~self.mapping['ItemID'].isin(missing)]
            print(f"Removed {count - len(self.mapping)} items from mapping. {len(self.mapping)} items remaining.")
        elif adjustMapping == 2:
            self.mapping = missing_df
            print(f"Replaced mapping with {len(missing)} items.")

        return missing_df

    def generate_embeddings(self):
        embedder, walkers, kg = self.return_rdf2vec_config(useBlazegraph=False)
        transformer = RDF2VecTransformer(
            embedder=embedder,
            walkers=walkers,
            verbose=2
        )

        if not os.path.exists(W2V_MODEL_LOC):
            os.makedirs(W2V_MODEL_LOC)

        if os.path.exists(W2V_MODEL_LOC + self.config.name + ".bin"):
            print("Loading existing model...")
            transformer.embedder.load(self.config.name)
        else:
            print("Fitting walks...")
            sentences = SentenceIterator(self.config.name, self.mapping["ItemID"].tolist())

            tic = time.perf_counter()
            transformer.embedder.fit(sentences, self.config.name, False)
            toc = time.perf_counter()

            print(f"Fitted walks ({toc - tic:0.4f}s)")

        entities = self.mapping['DBpediaURI'].tolist()
        item_ids = self.mapping['ItemID'].tolist()
        
        print("Transforming walks...")
        embeddings, literals = transformer.transform(kg, entities)

        del embedder
        del walkers
        del kg
        del transformer

        for index, item_id in enumerate(item_ids):
            if item_id is None or item_id == '' or embeddings[index] is None or len(embeddings[index]) == 0:
                print(f"Could not create embedding for item {item_id}.")
                continue
            
            if len(literals) == 0:
                self.save_embeddings_to_store(item_id, embeddings[index], []) # thread safe
            else:
                self.save_embeddings_to_store(item_id, embeddings[index], literals[index])

        self.save_embeddings_to_file()

    def save_embeddings_to_store(self, entity_id, embedding, literal):
        try:
            with self.lock:  # Ensure thread safety
                self.embedding_store[entity_id] = {'embeddings': embedding.tolist(), 'literals': literal}
        except Exception as e:
            print(f"An error occurred while saving to store: {e}")
            traceback.print_exc()

    def load_embeddings_from_store(self, entity_id):
        # Verify the type of the first entity in the store
        with self.lock:  # Ensure thread safety
            data = self.embedding_store.get(entity_id, {})
            if not data:
                return None, None
            else:
                return data.get('embeddings', []), data.get('literals', [])
        
    def save_embeddings_to_file(self):
        try:
            with self.fileLock:  # Ensure thread safety
                with open(EMBEDDING_STORE_LOC + self.config.name + ".json", "w+") as f:
                    json.dump(self.embedding_store, f)
        except Exception as e:
            print(f"An error occurred while saving embeddings to file: {e}")
            traceback.print_exc()

    def load_embeddings_from_file(self):
        # test if file exists
        if not os.path.isfile(EMBEDDING_STORE_LOC + self.config.name + ".json"):
            print(f"Embeddings file does not exist.")
            return False

        print("Loading embeddings from file...")
        with open(EMBEDDING_STORE_LOC + self.config.name + ".json", "r") as f:
            self.embedding_store = json.load(f)
        self.embedding_store = {int(k): v for k, v in self.embedding_store.items()}  # Convert keys to int
        print(f"Loaded {len(self.embedding_store)} embeddings from file.")
        return True

    # adjustMapping = 0 -> return missing
    # adjustMapping = 1 -> remove missing from mapping and return missing
    # adjustMapping = 2 -> replace mapping with missing and return missing
    def get_missing_embeddings(self, adjustMapping=0):
        correspondence = self.mapping['ItemID'].isin(list(map(int, self.embedding_store.keys())))
        missing = self.mapping[~correspondence]
        count = len(self.mapping)

        if adjustMapping == 1:
            self.mapping = self.mapping[correspondence]
            print(f"Removed {count - len(self.mapping)} items from mapping. {len(self.mapping)} items remaining.")
        elif adjustMapping == 2:
            self.mapping = missing
            print(f"Replaced mapping with {len(missing)} items.")

        return missing
    
    # def calculate_similarity(self):
    #     # Initialize an empty DataFrame for similarity
    #     self.similarity_df = pd.DataFrame(columns=['EntityID1', 'EntityID2', 'Similarity'])

    #     # Loop through each pair of entities to calculate similarity
    #     for i, row1 in self.mapping.iterrows():
    #         item1 = row1['ItemID']
    #         embeddings1, _ = self.load_from_redis(item1)
    #         for j, row2 in self.mapping.iterrows():
    #             if i <= j:  # Avoid redundant calculations
    #                 item2 = row2['ItemID']
    #                 embeddings2, _ = self.load_from_redis(item2)

    #                 # Calculate cosine similarity
    #                 sim_score = cosine_similarity([embeddings1], [embeddings2])[0][0]

    #                 # Append to DataFrame
    #                 self.similarity_df = self.similarity_df.append({
    #                     'EntityID1': item1,
    #                     'EntityID2': item2,
    #                     'Similarity': sim_score
    #                 }, ignore_index=True)

    # Helper functions for literals comparison
    def compare_literals(self, literals1, literals2):
        # Initialize literals similarity
        literals_sim_score = 0
        literals_count = len(literals1)

        # Ensure both literals arrays are of the same length
        if len(literals1) != len(literals2):
            print("Warning: Length of literals arrays do not match. Skipping literals similarity calculation.")
            return 0
        elif literals_count == 0:
            return 1

        # Calculate literals similarity
        for i in range(literals_count):
            literal1 = literals1[i]
            literal2 = literals2[i]
            if (isinstance(literal1, float) and math.isnan(literal1)) and (isinstance(literal2, float) and math.isnan(literal2)):
                literals_sim_score += 1
            elif isinstance(literal1, tuple) and isinstance(literal2, tuple):
                # Count common elements in both tuples
                literals_sim_score += len(set(literal1).intersection(set(literal2))) / max(len(literal1), len(literal2))
            elif (isinstance(literal1, float) and isinstance(literal2, float) and (not math.isnan(literal1)) and (not math.isnan(literal2))) \
                 or (isinstance(literal1, int) and isinstance(literal2, int)):
                # Continuous
                literals_sim_score += 1 - abs(literal1 - literal2) / max(literal1, literal2) # Continuous similarity
            elif isinstance(literal1, str) and isinstance(literal2, str):
                # Categorical
                literals_sim_score += 1 if literal1 == literal2 else 0
            else:
                # Unknown or mixed types
                literals_sim_score += 0.5

        return literals_sim_score / literals_count
            
    # def calculate_single_similarity(self, i, row1):
    #     similarities = []
    #     if 'ItemID' not in row1:
    #         item1 = row1[1]['ItemID']
    #     else:
    #         item1 = row1['ItemID']
    #     #embeddings1, literals1 = self.load_from_redis(item1)
    #     embeddings1, literals1 = self.load_embeddings(item1)

    #     for j, row2 in self.mapping.iterrows():
    #         if i <= j:  # Avoid redundant calculations
    #             item2 = row2['ItemID']
    #             #embeddings2, _literals2 = self.load_from_redis(item2)
    #             embeddings2, literals2 = self.load_embeddings(item2)

    #             # Calculate embedding similarity
    #             emb_sim_score = cosine_similarity([embeddings1], [embeddings2])[0][0]

    #             # Combine embedding and literals similarity
    #             #final_sim_score = 0.5 * emb_sim_score + 0.5 * literals_sim_score
    #             final_sim_score = emb_sim_score

    #             similarities.append({
    #                 'EntityID1': item1,
    #                 'EntityID2': item2,
    #                 'Similarity': final_sim_score
    #             })

    #     return similarities

    def calculate_single_similarity(self, row_tuple, include_literals=True):
        i, row1 = row_tuple
        item1 = row1['ItemID']
        embeddings1, literals1 = self.load_embeddings_from_store(item1)

        for j, row2 in self.mapping.iterrows():
            if i <= j:  # Avoid redundant calculations
                item2 = row2['ItemID']
                embeddings2, literals2 = self.load_embeddings_from_store(item2)

                # Calculate embedding similarity
                sim_score = cosine_similarity([embeddings1], [embeddings2])[0][0]

                # Calculate literals similarity
                if include_literals:
                    literals_sim_score = self.compare_literals(literals1, literals2)
                    sim_score = self.config.params.emb_weight * sim_score + self.config.params.lit_weight * literals_sim_score


                with self.lock:  # Ensure thread safety
                    self.similarity_df.loc[item1, item2] = sim_score
                    self.similarity_df.loc[item2, item1] = sim_score

    # def calculate_similarity(self):
    #     all_results = []
    #     with ThreadPoolExecutor() as executor:
    #         # Using tqdm for progress display
    #         for result in tqdm(executor.map(self.calculate_single_similarity, self.mapping.index, self.mapping.iterrows()), total=len(self.mapping)):
    #             with self.lock:  # Ensure thread safety
    #                 all_results.extend(result)

    #     self.similarity_df = pd.DataFrame(all_results)
    #     self.similarity_df.to_csv(SIMILARITY_LOC, index=False)

    def calculate_similarity(self, include_literals=True, saveToFile=True):
        print("Calculating similarity...")
        start = time.time()
        self.similarity_df = pd.DataFrame(index=self.mapping['ItemID'], columns=self.mapping['ItemID']).fillna(0.0)
        with ThreadPoolExecutor(max_workers=None) as executor:
            for _ in tqdm(executor.map(self.calculate_single_similarity, self.mapping.iterrows(), [include_literals] * len(self.mapping)), total=len(self.mapping)):
                pass
        end = time.time()
        if saveToFile:
            if not include_literals:
                self.similarity_df.to_csv(SIMILARITY_LOC + self.config.name + "_no_literals.csv", index=True)
            else:
                self.similarity_df.to_csv(SIMILARITY_LOC + self.config.name + "_with_literals.csv", index=True)
        print(f"Similarity matrix calculated in {end - start} seconds.")

    def load_similarity(self, include_literals=True):
        if not include_literals:
            path = SIMILARITY_LOC + self.config.name + "_no_literals.csv"
        else:
            path = SIMILARITY_LOC + self.config.name + "_with_literals.csv"
        
        if not os.path.isfile(path):
            print(f"Similarity file does not exist.")
            return False
        
        self.similarity_df = pd.read_csv(path, index_col=0)

        # columns to int
        self.similarity_df.columns = self.similarity_df.columns.astype(int)

        print(f"Loaded similarity matrix with {len(self.similarity_df)} items.")
        return True

    def calculate_common_items(self):
        data_item_ids = []
        for item_id, _ in self.embedding_store.items():
            data_item_ids.append(int(item_id))

        ratings_unique_items = np.sort(self.ratings['ItemID'].unique())

        self.common_items = np.intersect1d(ratings_unique_items, data_item_ids)

        print(f"Found {len(self.common_items)} items in common between ratings and item data.")

    def convert_ratings_to_binary(self, ratings_df):
        user_avg_rating = ratings_df.groupby('UserID')['Rating'].mean()

        new_ratings = []
        for index, row in tqdm(ratings_df.iterrows(), total=len(ratings_df), desc="Converting ratings to binary"):
            user_id = row['UserID']
            rating = row['Rating']

            # Fetch the average rating for the user
            avg_rating = user_avg_rating[user_id]     

            # Convert the rating to binary based on the user's average rating
            binary_rating = 1 if rating >= avg_rating else -1     

            # Append the new binary rating to the list
            new_ratings.append(binary_rating)

        # Replace the original 'Rating' column with the new binary ratings
        if self.config.params.convert_ratings_to_binary:
            ratings_df['Rating'] = new_ratings
        ratings_df['BinaryRating'] = new_ratings
        
        return ratings_df

    def runItemKNN(self, saveToFile=True, experiment_suffix="", methodology="allUnrated"): # methodology = allUnrated, ratedTestItems
        print("Running Item KNN")

        # Convert the ratings to binary
        train_ratings_df = self.convert_ratings_to_binary(self.train_ratings)
        test_ratings_df = self.convert_ratings_to_binary(self.test_ratings)

        train_ratings_dict = {}
        for row in tqdm(train_ratings_df.itertuples(), total=len(train_ratings_df), desc="Creating train ratings dict"):
            user_id = row[1]
            item_id = row[2]
            rating = row[3]

            if user_id not in train_ratings_dict:
                train_ratings_dict[user_id] = {}
            train_ratings_dict[user_id][item_id] = rating

        if methodology == "allUnrated": # else is ratedTestItems
            # Use all unrated items (that are not in the train set)
            users = self.ratings['UserID'].unique()
            allItems = set(self.mapping['ItemID'].unique())
            unrated_items = []

            for user in tqdm(users, total=len(users), desc="Generating unrated items"):
                # Items rated by the user in the test set
                test_rated_items = set(test_ratings_df[test_ratings_df['UserID'] == user]['ItemID'])
                
                # Items rated by the user in the train set
                train_rated_items = set(train_ratings_df[train_ratings_df['UserID'] == user]['ItemID'])
                
                # Unrated items for this user (not in test and train sets)
                user_unrated_items = allItems - test_rated_items - train_rated_items
                
                # Create rows for unrated items
                unrated_items.extend([(user, item, None, None) for item in user_unrated_items])

            # Create the "all unrated items" DataFrame
            considered = pd.DataFrame(unrated_items, columns=['UserID', 'ItemID', 'Rating', 'BinaryRating'])

            # Concatenate it with the test ratings DataFrame
            considered = pd.concat([considered, test_ratings_df[['UserID', 'ItemID', 'Rating', 'BinaryRating']]])
            considered.reset_index(drop=True, inplace=True)
        elif methodology == "ratedTestItems":
            # Use only the items that are in the test set
            considered = test_ratings_df
        else:
            print(f"Invalid methodology {methodology} specified.")
            exit()

        recommendation_scores = []
        for row in tqdm(considered.itertuples(), total=len(considered), desc="Calculating recommendation scores"):
            user_id = row[1]
            item_id = row[2]
            rating = row[3]
            binaryRating = row[4]

            if item_id not in self.similarity_df.columns:
                # Item not in similarity matrix
                continue

            if user_id in train_ratings_dict:
                previous_ratings = train_ratings_dict[user_id]
            else:
                previous_ratings = {}

            # Calculate the weighted average of the ratings using the similarity scores as weights
            weighted_sum = 0
            count = 0
            for item, prev_rating in previous_ratings.items():
                if item not in self.similarity_df.columns:
                    # Item not in similarity matrix
                    continue

                similarity = self.similarity_df.loc[item_id, item]
                weighted_sum += prev_rating * similarity
                count += 1

            if count == 0:
                # No common items. Use item KNN
                continue

            recommendation_score = weighted_sum / count

            #self.recommendation_scores.loc[len(self.recommendation_scores)] = [user_id, item_id, int(rating), recommendation_score, count, weighted_sum]
            recommendation_scores.append([user_id, item_id, rating, binaryRating, recommendation_score, count, weighted_sum])

        self.recommendation_scores = pd.DataFrame(data=recommendation_scores, columns=['UserID', 'ItemID', 'Rating', 'BinaryRating', 'RecommendationScore', 'HistoryCount', 'WeightedSum'])

        print(f"Recommendation score stats:")
        print(self.recommendation_scores['RecommendationScore'].describe())

        if saveToFile:
            print("Saving recommendation scores to file...")
            self.recommendation_scores.to_csv(RECOMMENDATION_SCORES_LOC + self.config.name + experiment_suffix + ".csv", index=False)
            print("Successfully saved recommendation scores to file.")

    def load_recommendation_scores(self, experiment_suffix=""):
        path = RECOMMENDATION_SCORES_LOC + self.config.name + experiment_suffix + ".csv"

        if not os.path.isfile(path):
            print(f"Predicted scores file does not exist.")
            return False
        
        with open(path, "r") as f:
            self.recommendation_scores = pd.read_csv(f)

        if len(self.recommendation_scores) == 0:
            print(f"Predicted scores file is empty.")
            return False
        return True

    # def evaluate(self):
    #     # Calculating metrics
    #     y_true = self.recommendation_scores['Rating']
    #     y_pred = self.recommendation_scores['RecommendationScore']

    #     precision = precision_score(y_true, y_pred, average='weighted')
    #     recall = recall_score(y_true, y_pred, average='weighted')
    #     f1 = f1_score(y_true, y_pred, average='weighted')
    #     hits = sum(y_true == y_pred)

    #     print(f"Precision: {precision}")
    #     print(f"Recall: {recall}")
    #     print(f"F1: {f1}")
    #     print(f"Hits: {hits}")

    def evaluate_at_N(self, N, saveToFile=True, experiment_suffix=""):
        # Sort the DataFrame by 'RecommendationScore' in descending order for each user
        sorted_ratings = self.recommendation_scores.sort_values(by=['UserID', 'RecommendationScore'], ascending=[True, False])
        
        # Get unique user IDs
        user_ids = sorted_ratings['UserID'].unique()

        print(f"Calculating metrics for N={N}...")
        precisions = []
        recalls = []
        f1s = []
        hit_list = []
        ndcgs = []
        skipped_users = 0
        
        for user_id in tqdm(user_ids, total=len(user_ids), desc="Calculating metrics"):
            # Get the top-N items recommended for this user
            top_n_items = sorted_ratings.loc[sorted_ratings['UserID'] == user_id].head(N)

            if len(top_n_items) != N:
                skipped_users += 1
                continue
            
            tp = 0
            fp = 0
            hit = 0

            total_relevant_items = len(sorted_ratings.loc[(sorted_ratings['UserID'] == user_id) & (sorted_ratings['BinaryRating'] == 1)])

            for _, recommendation in top_n_items.iterrows():
                if recommendation["BinaryRating"] == 1:
                    tp += 1
                    hit = 1
                else:
                    fp += 1
            
            fn = total_relevant_items - tp
            
            # Calculate precision and recall for this user
            if tp + fp == 0:
                precision = 0
            else:
                precision = tp / (tp + fp)
            
            if tp + fn == 0:
                recall = 0
            else:
                recall = tp / (tp + fn)

            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

            if self.config.dataset == "Movielens1M":
                top_n_items['Rating'].replace({np.nan: 3}, inplace=True)
            elif self.config.dataset == "Librarything":
                top_n_items['Rating'].replace({np.nan: 5}, inplace=True)
            elif self.config.dataset == "Lastfm":
                user_ratings = self.recommendation_scores.loc[self.recommendation_scores['UserID'] == user_id, 'Rating']
                user_ratings = user_ratings[~np.isnan(user_ratings)]
                user_neutral_value = np.median(user_ratings)
                top_n_items['Rating'].replace({np.nan: user_neutral_value}, inplace=True)

            #Calculate DCG for this user
            dcg = np.sum((2 ** top_n_items['Rating'] - 1) / np.log2(np.arange(2, N + 2)))
            
            # Calculate IDCG for this user
            idcg = np.sum((2**np.sort(top_n_items['Rating'])[::-1] - 1) / np.log2(np.arange(2, N + 2)))

            # Calculate nDCG for this user
            if idcg == 0:
                ndcg = 0
            else:
                ndcg = dcg / idcg
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            hit_list.append(hit)
            ndcgs.append(ndcg)
        
        # Calculate the average precision and recall across all users
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_f1 = np.mean(f1s)
        avg_hits = np.mean(hit_list)
        avg_ndcg = np.mean(ndcgs)

        print(f"Precision@{N}: {avg_precision}")
        print(f"Recall@{N}: {avg_recall}")
        print(f"F1@{N}: {avg_f1}")
        print(f"Hit@{N}: {avg_hits}")
        print(f"nDCG@{N}: {avg_ndcg}")
        print(f"Skipped users@{N}: {skipped_users}")

        self.results[self.config.name + experiment_suffix + '_@' + str(N)] = {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
            "hits": avg_hits,
            "ndcg": avg_ndcg,
            "skipped": skipped_users
        }

        if saveToFile:
            self.save_results()

        return avg_precision, avg_recall, avg_f1, avg_hits, avg_ndcg, skipped_users

    def evaluate(self, saveToFile=True, experiment_suffix=""):
        # Create empty lists to store results for varying N
        precision_at_N = []
        recall_at_N = []
        f1_at_N = []
        hits_at_N = []
        skipped_at_N = []
        ndcg_at_N = []
        
        # Sort the DataFrame by 'RecommendationScore' in descending order for each user
        sorted_ratings = self.recommendation_scores.sort_values(by=['UserID', 'RecommendationScore'], ascending=[True, False])
        
        # Get unique user IDs
        user_ids = sorted_ratings['UserID'].unique()
        
        for N in range(1, 21):  # Varying N in the interval [1, ..., 20]
            print(f"Calculating metrics for N={N}...")
            precisions = []
            recalls = []
            f1s = []
            hit_list = []
            ndcgs = []
            skipped_users = 0
            
            for user_id in user_ids:
                # Get the top-N items recommended for this user
                top_n_items = sorted_ratings.loc[sorted_ratings['BinaryRating'] == user_id].head(N)

                if len(top_n_items) != N:
                    skipped_users += 1
                    continue
                
                tp = 0
                fp = 0
                hit = 0

                total_relevant_items = len(sorted_ratings.loc[(sorted_ratings['UserID'] == user_id) & (sorted_ratings['Rating'] == 1)])

                for _, recommendation in top_n_items.iterrows():
                    if recommendation["Rating"] == 1:
                        tp += 1
                        hit = 1
                    else:
                        fp += 1
                
                fn = total_relevant_items - tp
                
                # Calculate precision and recall for this user
                if tp + fp == 0:
                    precision = 0
                else:
                    precision = tp / (tp + fp)
                
                if tp + fn == 0:
                    recall = 0
                else:
                    recall = tp / (tp + fn)

                if precision + recall == 0:
                    f1 = 0
                else:
                    f1 = 2 * (precision * recall) / (precision + recall)

                #Calculate DCG for this user
                dcg = np.sum(top_n_items['Rating'] / np.log2(np.arange(2, N + 2)))  # 2, 3, ..., N+1 for log base
                
                # Calculate IDCG for this user
                idcg = np.sum(sorted(top_n_items['Rating'], reverse=True) / np.log2(np.arange(2, N + 2)))

                # Calculate nDCG for this user
                if idcg == 0:
                    ndcg = 0
                else:
                    ndcg = dcg / idcg
                
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
                hit_list.append(hit)
                ndcgs.append(ndcg)
            
            # Calculate the average precision and recall across all users
            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)
            avg_f1 = np.mean(f1s)
            avg_hits = np.mean(hit_list)
            avg_ndcg = np.mean(ndcgs)
            
            precision_at_N.append(avg_precision)
            recall_at_N.append(avg_recall)
            f1_at_N.append(avg_f1)
            hits_at_N.append(avg_hits)
            ndcg_at_N.append(avg_ndcg)
            skipped_at_N.append(skipped_users)
            
            print(f"Precision@{N}: {avg_precision}")
            print(f"Recall@{N}: {avg_recall}")
            print(f"F1@{N}: {avg_f1}")
            print(f"Hit@{N}: {avg_hits}")
            print(f"nDCG@{N}: {avg_ndcg}")
            print(f"Skipped users@{N}: {skipped_users}")

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.scatter(recall_at_N, precision_at_N, c='blue', marker='o', label='Data Points')
        plt.title('Precision-Recall Curve for Varying N')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        # Annotate each point with its corresponding N value
        for i, N in enumerate(range(1, 21)):
            plt.annotate(f"N={N}", (recall_at_N[i], precision_at_N[i]), textcoords="offset points", xytext=(0,10), ha='center')
        
        # Save the plot
        fig_log = FIGURES_LOC + self.config.name + experiment_suffix + ".png"
        plt.savefig(fig_log)
        
        self.results[self.config.name + experiment_suffix] = {
            "precision": precision_at_N,
            "recall": recall_at_N,
            "f1": f1_at_N,
            "hits": hits_at_N,
            "ndcg": ndcg_at_N,
            "skipped": skipped_at_N,
            "fig_location": fig_log
        }

        if saveToFile:
            self.save_results()
    
    def load_final_results(self):
        try: 
            with open(EVAL_FILE_LOC, "r") as f:
                self.results = json.load(f)
        except:
            self.results = {}

    def save_results(self):
        with open(EVAL_FILE_LOC, "w+") as f:
            json.dump(self.results, f)

    def set_default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        raise TypeError
    
    def save_config(self):
        with open(CONFIGS_LOC + self.config["name"] + ".json", "w+") as f:
            json.dump(self.config, f, default=self.set_default)

    def load_config(self, name):
        with open(CONFIGS_LOC + name + ".json", "r") as f:
            self.config = json.load(f)


    def vectors_from_uri(self, uri):
        endpoint = SPARQLEndpoint("http://dbpedia.org/sparql")
        endpoint.connectDBpedia()

        item = endpoint.describe_resource(uri)
        endpoint.write_resource_to_file(uri, "asturias.ttl")
    
def embedding_test(recommender):
    filtered = recommender.mapping[recommender.mapping['ItemID'].isin([1, 3114, 3917])]
    recommender.mapping = filtered
    start = time.time()
    recommender.generate_and_save_embeddings(saveToFile=False)
    end = time.time()
    print(f"Time taken: {end - start}")
    #recommender.load_embeddings_from_file()

    recommender.similarity_df = pd.DataFrame(index=recommender.mapping['ItemID'], columns=recommender.mapping['ItemID']).fillna(0.0)
    for row_tuple in filtered.iterrows():
        recommender.calculate_single_similarity(row_tuple)
    
    print(f"1:2 -> {recommender.similarity_df.loc[1, 3114]}")
    print(f"1:3 -> {recommender.similarity_df.loc[1, 3917]}")
    print(f"2:3 -> {recommender.similarity_df.loc[3114, 3917]}")

    exit()

def similarity_test(recommender):
    # recommender.load_embeddings_from_file()
    # recommender.get_missing(adjustMapping=1)
    # recommender.calculate_similarity(False)
    # recommender.similarity_df.to_csv("data/results/sim_test.csv", index=False)

    recommender.similarity_df = pd.read_csv("data/results/sim_test.csv")

    # Remove the diagonal elements for a true measure of similarity between different items
    without_diag = recommender.similarity_df[~np.eye(recommender.similarity_df.shape[0], dtype=bool)]

    min_similarity = without_diag.min().min()
    max_similarity = without_diag.max().max()
    avg_similarity = without_diag.mean().mean()
    median_similarity = without_diag.median().median()

    print(f"Min similarity: {min_similarity}")
    print(f"Max similarity: {max_similarity}")
    print(f"Avg similarity: {avg_similarity}")
    print(f"Median similarity: {median_similarity}")

    similarity_values = without_diag[~np.isnan(without_diag)]

    #similarity_values = np.array(similarity_values).flatten()  # Flatten the array to 1D

    plt.figure(figsize=(10, 6))
    plt.hist(similarity_values, bins=10, density=True, alpha=0.7)

    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig("data/results/similarity_distribution.png")

    exit()

def random_user_exp(recommender):
    # Select a random user
    user_id = np.random.choice(recommender.ratings['UserID'].unique())

    # Get the ratings for the selected user
    user_ratings = recommender.ratings.loc[recommender.ratings['UserID'] == user_id]

    # Get the items that the user has rated
    user_items = user_ratings['ItemID'].unique()

    # Replace the mapping
    recommender.mapping = recommender.mapping[recommender.mapping['ItemID'].isin(user_items)]

    print(f"Selected user {user_id} who has rated {len(user_items)} items.")

    # Usual pipeline
    recommender.launch_blazegraph(port=9999) #url="http://10.20.56.36:19999/blazegraph/"

    print(f"Generating embeddings for {len(recommender.mapping)} items...")
    recommender.generate_and_save_embeddings(saveToFile=False)
    print(f"Embeddings generated for {len(recommender.mapping)} items.")

    recommender.calculate_similarity(saveToFile=False)
    print(f"Similarity calculated. Stats:")
    print(recommender.similarity_df.describe())

    recommender.runItemKNN(saveToFile=False)

    recommender.evaluate(saveToFile=False)

# return the K-nearest neighbors of a movie
def KNN_test(recommender, k=10, movie="Batman (1989)", compare_against=["On the Ropes (1999)"]):
    # Find movie ids
    mapping = recommender.mapping
    movie_id = mapping.loc[mapping['Title'] == movie, 'ItemID'].values[0]
    movie_uri = mapping.loc[mapping['Title'] == movie, 'DBpediaURI'].values[0]

    if compare_against:
        compare_against_ids = mapping.loc[mapping['Title'].isin(compare_against), 'ItemID'].values.tolist()
        compare_against_ids = [int(i) for i in compare_against_ids]
        compare_against_uris = mapping.loc[mapping['Title'].isin(compare_against), 'DBpediaURI'].values.tolist()

        entities = [movie_uri] + compare_against_uris

        knowledge_graph = KG(
            "http://131.211.32.87:19999/blazegraph/" + BLAZEGRAPH_NAMESPACE,
            skip_predicates={
                "www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "http://xmlns.com/foaf/0.1/depiction",
                "http://dbpedia.org/property/wikiPageUsesTemplate",
                "http://dbpedia.org/ontology/thumbnail",
                "http://dbpedia.org/property/wordnet_type",
                "http://www.w3.org/ns/prov#wasDerivedFrom",
                "http://xmlns.com/foaf/0.1/isPrimaryTopicOf",
                "http://dbpedia.org/ontology/wikiPageDisambiguates",
                "http://xmlns.com/foaf/0.1/primaryTopic",
                "http://xmlns.com/foaf/0.1/isPrimaryTopicOf",
                "http://dbpedia.org/property/wikiPageExternalLink",
                "http://dbpedia.org/property/wikiPageID",
                "http://dbpedia.org/property/wikiPageRevisionID",
                "http://www.w3.org/2002/07/owl#sameAs",
                "http://dbpedia.org/ontology/wikiPageWikiLink",
                "http://schema.org/sameAs",
                "http://purl.org/linguistics/gold/hypernym",
                "http://www.w3.org/2000/01/rdf-schema#seeAlso",
                "http://www.w3.org/2002/07/owl#differentFrom",
            },
            skip_verify=True,
            mul_req=True
        )

        transformer = RDF2VecTransformer(
            Word2Vec(epochs=10, 
                    sg=1, # 0 for CBOW, 1 for skip-gram
                    vector_size=200, 
                    window=5, 
                    workers=4, 
                    negative=25),
            walkers=[RandomWalker(4, 200, with_reverse=True, n_jobs=10)],
            verbose=1
        )

        #embeddings, _ = transformer.fit_transform(knowledge_graph, entities)
        transformer._is_extract_walks_literals = True
        walks = transformer.get_walks(knowledge_graph, entities)

        tic = time.perf_counter()
        transformer.embedder.fit(walks, False)
        toc = time.perf_counter()

        n_walks = sum([len(entity_walks) for entity_walks in walks])
        print(f"Fitted {n_walks} walks ({toc - tic:0.4f}s)")
        
        embeddings, _ = transformer.transform(knowledge_graph, entities)

        movie_embedding = embeddings[0]
        compare_against_embeddings = embeddings[1:]

        # Calculate cosine similarity
        similarity_scores = cosine_similarity([movie_embedding], compare_against_embeddings)[0]

        # to pd
        similarity_scores = pd.Series(similarity_scores, index=compare_against_ids)
    else:
        recommender.load_similarity(include_literals=False)

        # Get the similarity scores of the movie with all other movies
        similarity_scores = recommender.similarity_df.loc[movie_id]

        # Remove the similarity score of the movie with itself
        similarity_scores.drop(str(movie_id), inplace=True)

    # Sort the scores in descending order
    similarity_scores.sort_values(ascending=False, inplace=True)

    # Get the top-K similar movies
    top_k_similar_movies = similarity_scores.iloc[:k+1]

    # Get the indices of the top-K similar movies
    top_k_similar_movies_indices = top_k_similar_movies.index.values

    # to int
    top_k_similar_movies_indices = [int(i) for i in top_k_similar_movies_indices]

    # Get the titles of the top-K similar movies
    top_k_similar_movies_titles = recommender.mapping.loc[recommender.mapping['ItemID'].isin(top_k_similar_movies_indices), 'Title']

    # Print the results
    print(f"Top-{k} similar movies to {movie}:")
    print(top_k_similar_movies_titles.values)

    exit()


if __name__ == "__main__":
    config = {
        "name": "movielens_random_uniform_200w_200v_4d_rev", # movielens_random_uniform_200w_200v_4d_rev, lastfm_random_uniform_200w_200v_4d_rev
        "dataset": "Movielens1M", # Movielens1M, Librarything, Lastfm
        "params": {
            "walker": "RandomWalker",
            "walkerConfig": {
                "max_depth": 4,
                "max_walks": 200,
                "with_reverse": True,
                "sampler": "UniformSampler",
                "samplerConfig": {
                    "split": True,
                    "inverse": False,
                    "alpha": 0.85,
                }
            },
            "embedder": "Word2Vec",
            "embedderConfig": {
                "epochs": 5,
                "sg": 1, # 0 for CBOW, 1 for Skip-gram
                "vector_size": 200,
                "window": 5,
                "negative": 25,
            },
            "skip_predicates": {
                "www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "http://xmlns.com/foaf/0.1/depiction",
                "http://dbpedia.org/property/wikiPageUsesTemplate",
                "http://dbpedia.org/ontology/thumbnail",
                "http://dbpedia.org/property/wordnet_type",
                "http://www.w3.org/ns/prov#wasDerivedFrom",
                "http://xmlns.com/foaf/0.1/isPrimaryTopicOf",
                "http://dbpedia.org/ontology/wikiPageDisambiguates",
                "http://xmlns.com/foaf/0.1/primaryTopic",
                "http://xmlns.com/foaf/0.1/isPrimaryTopicOf",
                "http://dbpedia.org/property/wikiPageExternalLink",
                "http://dbpedia.org/property/wikiPageID",
                "http://dbpedia.org/property/wikiPageRevisionID",
                "http://www.w3.org/2002/07/owl#sameAs",
                "http://dbpedia.org/ontology/wikiPageWikiLink",
                "http://schema.org/sameAs",
                "http://purl.org/linguistics/gold/hypernym",
                "http://www.w3.org/2000/01/rdf-schema#seeAlso",
                "http://www.w3.org/2002/07/owl#differentFrom",
            },
            "literals": [
                [
                    "http://dbpedia.org/ontology/wikiPageWikiLink",
                    "http://www.w3.org/2000/01/rdf-schema#label",
                ],
                ["http://dbpedia.org/ontology/budget"],
                ["http://dbpedia.org/ontology/gross"],
                ["http://dbpedia.org/property/writer"],
                ["http://dbpedia.org/property/cinematography"],
                ["http://dbpedia.org/property/country"],
                ["http://dbpedia.org/property/language"],
                ["http://dbpedia.org/property/producer"],
                ["http://dbpedia.org/property/studio"]
            ],
            "emb_weight": 1,
            "lit_weight": 0,
            "convert_ratings_to_binary": False,
        }
    }

    recommender = GPRS(config)
    recommender.save_config()

    recommender.load_mapping()

    # embedding_test(recommender)
    # similarity_test(recommender)
    # random_user_exp(recommender)
    # KNN_test(recommender, 10, "Batman (1989)")

    recommender.load_ratings()
    recommender.data_preprocessing()

    # recommender.mapping = recommender.mapping.iloc[:10]
    # tenRows = recommender.mapping.iloc[1]
    # recommender.process_single_item(firstRow['ItemID'], firstRow['DBpediaURI'])

    recommender.extract_walks(batchSize=6, bgURL=None) # "http://131.211.32.87:19999/blazegraph/"
    
    if not recommender.load_embeddings_from_file():
        recommender.generate_embeddings()

    if not recommender.load_similarity(include_literals=False):
        recommender.calculate_similarity(include_literals=False)

    if not recommender.load_recommendation_scores(experiment_suffix="_no_literals"):
        recommender.runItemKNN(experiment_suffix="_no_literals")

    recommender.load_final_results()

    #recommender.evaluate(experiment_suffix="_no_literals")
    recommender.evaluate_at_N(10, experiment_suffix="_no_literals")
    