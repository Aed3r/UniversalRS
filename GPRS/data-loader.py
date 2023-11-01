"""
item.py

NOT USED ANYMORE

This class is used to load data from the data sources and to transform it into a format that can be used by the recommender system.

Author: Gustav Hubert
"""

import csv
import os
from database import DatabaseManager
from item import Item
from tqdm import tqdm
import logging

class DataLoader:
    """Loads data from CSV files and makes them available to a recommender system."""

    MOVIELENS_LOC = os.path.join('.', 'data', 'ml-25m')

    def __init__(self, db_manager=DatabaseManager()):
        """
        Initialize a new DataLoader instance.

        Args:
            db_manager (DatabaseManager): The instance of the DatabaseManager class to use for storing the data.
        """
        self.db_manager = db_manager

    def load_csv(self, source, type, file_path, length=None):
        """
        Load data from a CSV file and store it in the database.

        Args:
            file_path (str): The path to the CSV file.
            source (str): The source of the data.
            type (str): The type of the data.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in tqdm(csv_reader, desc="Importing items from source '" + source +"'", ncols=150, total=length):
                item = {"title": row['title'], "source": source, "type": type}
                # Add all columns except 'title' to the tags
                for column in csv_reader.fieldnames:
                    if column != 'title':
                        item[column] = row[column]
                self.db_manager.insert('items', item)

    def load_csv_supplement(self, join_field, file_path, source, length=None, resumeFrom=0):
        """
        Load data from a CSV file and store it in the database under the item that matches the join_field. The source file shoud contain the join_field as well as any other columns that should be added to the item.

        Args:
            join_field (str): The field to use to match the items.
            file_path (str): The path to the CSV file.
            source (str): The source of the data. Used as the name of the field in the item.
            length (int, optional): The number of rows in the CSV file used for the progress bar. Defaults to None.
            resumeFrom (int, optional): The number of rows to skip before starting to import. Defaults to 0.
        """

        failCount = 0
        sourceLower = source.lower()
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in tqdm(csv_reader, desc="Importing supplement items from source '" + source + "'", ncols=150, total=length):
                if resumeFrom > 0:
                    resumeFrom -= 1
                    continue

                # Find the item that matches the join_field
                item = self.db_manager.get_one('items', {join_field: row[join_field]})
                if item is not None:
                    supplement = {}
                    # Add all columns except the join field to the tags
                    for column in csv_reader.fieldnames:
                        if column != join_field:
                            supplement[column] = row[column]

                    # Check if the item already has the supplement as a field and update it accordingly
                    if sourceLower not in item:
                        result = self.db_manager.update_one('items', {join_field: row[join_field]}, {"$set": {sourceLower: [supplement]}})
                    else:
                        result = self.db_manager.update_one('items', {join_field: row[join_field]}, {"$push": {sourceLower: supplement}})
                    
                    if result.matched_count == 0:
                        failCount += 1
        
        if failCount > 0:
            logging.warning("WARNING: " + str(failCount) + " items could not be matched to an existing item.")


    def load_movieLens(self):
        """Load the MovieLens dataset into the database."""

        # Load the base data
        #self.load_csv('MovieLens', 'movie', os.path.join(DataLoader.MOVIELENS_LOC, 'movies.csv'), length=62423)

        # Create an index on the ID field
        #self.db_manager.create_index('items', 'movieId', unique=True)

        # Load the ratings data
        #self.load_csv_supplement('movieId', os.path.join(DataLoader.MOVIELENS_LOC, 'ratings.csv'), 'ratings', length=25000095, resumeFrom=824356)

        # Load the tags data
        self.load_csv_supplement('movieId', os.path.join(DataLoader.MOVIELENS_LOC, 'tags.csv'), 'tags', length=1093360)
        

if __name__ == '__main__':
    d = DataLoader()    
    d.load_movieLens()