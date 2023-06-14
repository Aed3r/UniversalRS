"""
database.py

This module defines the `DatabaseManager` class, which is used to manage the database. It provides methods to insert, update, and delete items from the database, as well as methods to import and export the database.

Author: Gustav Hubert
"""

import os
import time
from pymongo import MongoClient
import json
from tqdm import tqdm
from bson.json_util import dumps
import logging

class DatabaseManager:
    """Connects to the database and provides methods to insert, update, and delete items from the database, as well as methods to import and export the database."""

    COLLECTIONS = ['items', 'test'] # The collections used by the system
    VALIDATION_SCHEMA_SRC = "./GPRS/db_schema.json"

    def __init__(self, test=False):
        """Initialize a new DatabaseManager instance."""
        if test:
            self._db = MongoClient("mongodb://localhost:27017/")['GPRS_test']
        else:
            self._db = MongoClient("mongodb://localhost:27017/")['GPRS']
        self._schemaValidated = False # Does the validation schema of the db correspond to the one in db_schema.json?
        self._validationSchema = None # The validation schema of the db
        #self._basicDataLoaded = False # Is the basic data loaded into the db?
        
        # Load the validation schema
        self._load_validation_schema()

        # Create the collection
        self.create_collections()

    #### Creation / Validation / Destruction ####
        
    def _load_validation_schema(self):
        """Load the validation schema from db_schema.json."""
        try:
            with open(DatabaseManager.VALIDATION_SCHEMA_SRC, 'r') as j:
                self._validationSchema = json.load(j)
        except FileNotFoundError:
            logging.error('db_schema.json not found. Please update the path in src/database.py')
            exit(1)

    def create_collections(self):
        """Create the collections with the validation schema if they don't exist."""
        for collection in DatabaseManager.COLLECTIONS:
            # Check if the collection already exists
            if not collection in self._db.list_collection_names():
                self._db.create_collection(collection, validator=self._validationSchema)
                self._schemaValidated = True
                return
        
        self._validateSchema() # No need for check as the collection was just created

    def _validateSchema(self):
        """Validate the schema of the collections with the one in db_schema.json."""
        if self._schemaValidated:
            return True
        
        if self._validationSchema is None:
            self._load_validation_schema()

        # Get the schema of the db
        for collection in DatabaseManager.COLLECTIONS:
            current_schema = self._db.get_collection(collection).options().get('validator')

            # Compare the two schemas
            if not self._validationSchema == current_schema:
                return False
        
        self._schemaValidated = True
        return True
    
    def drop_collection(self, collectionName):
        """Drop the given collection.

        Args:
            collectionName (str): The name of the collection to drop.
        """
        self._db.drop_collection(collectionName)
        self._schemaValidated = False

    def drop_collections(self):
        """Drop all collections."""
        for collectionName in tqdm(DatabaseManager.COLLECTIONS, desc="Dropping collections", ncols=150):
            self.drop_collection(collectionName)
    
    #### Input / Output ####

    def insert(self, collectionName, data):
        """Insert a single item into the given collection.
        
        Args:
            collectionName (str): The name of the collection to insert the item into.
            data (dict): The data to insert.

        Returns:
            pymongo.results.InsertOneResult: The result of the insertion.
        """
        try:
            result = self._db[collectionName].insert_one(data)
        except:
            logging.error("Error inserting '{}' into collection {}".format(data['_id'], collectionName))

        return result

    def insert_all(self, collectionName, data):
        """"Insert multiple items into the given collection.
        
        Args:
            collectionName (str): The name of the collection to insert the items into.
            data (list): The data to insert.
        """
        for d in tqdm(data, desc="Inserting items into the database", ncols=150):
            self.insert(collectionName, d)

    def insert_id(self, collectionName, id, data):
        """Insert a single item with the given id into the given collection.
        
        Args:
            collectionName (str): The name of the collection to insert the item into.
            id (str): The id of the item.
            data (dict): The data to insert.

        Returns:
            pymongo.results.InsertOneResult: The result of the insertion.
        """
        data['_id'] = id
        return self.insert(collectionName, data)

    def get_all(self, collectionName):
        """Get all items from the given collection.

        Args:
            collectionName (str): The name of the collection to get the items from.
        """
        return self._db[collectionName].find({})
    
    def get(self, collectionName, query, filter={}, limit=None):
        """Get all items from the given collection that match the given query.

        Args:
            collectionName (str): The name of the collection to get the items from.
            query (dict): The query to match.
            filter (dict, optional): The filter to apply. Defaults to {}.
            limit (int, optional): The maximum number of items to return. Defaults to None.

        Returns:
            pymongo.cursor.Cursor: The cursor to the items that match the query.
        """
        if limit is None:
            return self._db[collectionName].find(query, filter)
        else:
            return self._db[collectionName].find(query, filter).limit(limit)
        
    def get_one(self, collectionName, query, filter={}):
        """Get the first item from the given collection that matches the given query.

        Args:
            collectionName (str): The name of the collection to get the item from.
            query (dict): The query to match.
            filter (dict, optional): The filter to apply. Defaults to {}.

        Returns:
            dict: The item that matches the query.
        """
        return self._db[collectionName].find_one(query, filter)
    
    def update(self, collectionName, query, update):
        """Update all items from the given collection that match the given query.

        Args:
            collectionName (str): The name of the collection to update the items in.
            query (dict): The query to match.
            update (dict): The update to apply.

        Returns:
            pymongo.results.UpdateResult: The result of the update.
        """
        return self._db[collectionName].update_many(query, update)

    def update_one(self, collectionName, query, update):
        """Update the first item from the given collection that matches the given query.

        Args:
            collectionName (str): The name of the collection to update the item in.
            query (dict): The query to match.
            update (dict): The update to apply.

        Returns:
            pymongo.results.UpdateResult: The result of the update.
        """
        return self._db[collectionName].update_one(query, update)
        
    #### Extras ####
    
    def get_collection_names(self):
        """Get the names of all collections."""
        return DatabaseManager.COLLECTIONS
    
    def get_current_collection_names(self):
        """Get the names of all collections in the database."""
        return self._db.list_collection_names()
    
    def get_collection_count(self, collectionName):
        """Get the number of items in the given collection.

        Args:
            collectionName (str): The name of the collection to get the number of items from.
        """
        return self._db[collectionName].count_documents({})
    
    def create_index(self, collectionName, fieldName, unique=False):
        """Create an index on the given field in the given collection.

        Args:
            collectionName (str): The name of the collection to create the index in.
            fieldName (str): The field to create the index on.
            unique (bool, optional): Whether the index should be unique. Defaults to False.
        """
        self._db[collectionName].create_index(fieldName, name=fieldName + "_index", unique=unique)

    #### Import / Export ####

    def export_collection(self, path, collectionName):
        """Export the named collection to a json file.
        
        Args:
            path (str): The path to the file to export to.
            collectionName (str): The name of the collection to export.
        """
        start = time.time()
        cursor = self._db[collectionName].find({})
        n = self.get_collection_count(collectionName)
        i = 0
        with open(os.path.join(path, collectionName + ".json"), 'w') as f:
            f.write('[')
            for item in tqdm(cursor, desc="Exporting collection {}".format(collectionName), ncols=150, total=n):
                f.write(dumps(item))
                if i != n - 1:
                    f.write(',')
                i += 1
            f.write(']')
        logging.info("Collection '{}' exported successfully. (Took {:.2f} seconds)".format(collectionName, time.time() - start))

    def export_collections(self, path):
        """Export all collections to a json file.
        
        Args:
            path (str): The path to the file to export to.
        """
        logging.info("Exporting collections to {}...".format(path))
        start = time.time()
        for collectionName in DatabaseManager.COLLECTIONS:
            self.export_collection(path, collectionName)
        logging.info("All collections exported successfully. (Took {:.2f} seconds)".format(time.time() - start))

    def import_collection(self, path, collectionName):
        """Import the named collection from a json file.

        Args:
            path (str): The path to the file to import from.
            collectionName (str): The name of the collection to import.
        """
        logging.info("Importing collection '{}' from {}...".format(collectionName, path))
        start = time.time()
        with open(path, 'r') as f:
            data = json.load(f)
        self.insert_all(data, collectionName)
        logging.info("Collection '{}' imported successfully. (Took {:.2f} seconds)".format(collectionName, time.time() - start))

    def import_collections(self, path):
        """Import all collections from a json file.

        Args:
            path (str): The path to the file to import from.
        """
        logging.info("Importing collections from {}...".format(path))
        start = time.time()
        for collectionName in DatabaseManager.COLLECTIONS:
            self.import_collection(os.path.join(path, collectionName + ".json"), collectionName)
        logging.info("All collections imported successfully. (Took {:.2f} seconds)".format(time.time() - start))
