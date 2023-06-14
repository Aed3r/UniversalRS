"""
test_database.py

This file contains the unit tests for the DatabaseManager class.

Author: Gustav Hubert
"""

import os
import unittest
from unittest.mock import MagicMock
import pymongo
from GPRS.database import DatabaseManager

class DatabaseManagerTest(unittest.TestCase):
    """Test the DatabaseManager class."""

    def setUp(self):
        """Set up the test environment."""
        self.dbm = DatabaseManager(test=True)

    def tearDown(self):
        """Clean up the test collections after each test."""
        self.dbm.drop_collections()

    def test_create_collections(self):
        """Test the creation of collections."""
        # Ensure that the collections are created with the validation schema
        self.dbm.create_collections()

        for collection in self.dbm.get_current_collection_names():
            self.assertTrue(collection in self.dbm.get_collection_names())
        
        self.assertTrue(self.dbm._validateSchema())

    def test_insert(self):
        """Test the insertion of a single item."""
        collection_name = 'test'
        data = {'_id': 1, 'name': 'Item 1'}

        # Insert the data into the collection
        self.dbm.insert(collection_name, data)

        # Check if the data exists in the collection
        collection = self.dbm._db[collection_name]
        inserted_data = collection.find_one({'_id': 1})
        self.assertIsNotNone(inserted_data)
        self.assertEqual(data['name'], inserted_data['name'])

    def test_insert_id(self):
        collection_name = 'test'
        item_id = '12345'
        item_data = {'name': 'Test Item', 'description': 'This is a test item.'}

        result = self.dbm.insert_id(collection_name, item_id, item_data)

        self.assertEqual(result.inserted_id, item_id)

        collection_count = self.dbm.get_collection_count(collection_name)
        self.assertEqual(collection_count, 1)

        inserted_item = list(self.dbm.get(collection_name, {'_id': item_id}))[0]
        self.assertIsNotNone(inserted_item)
        self.assertEqual(inserted_item['_id'], item_id)
        self.assertEqual(inserted_item['name'], item_data['name'])
        self.assertEqual(inserted_item['description'], item_data['description'])

    def test_get_all(self):
        """Test retrieving all items from a collection."""
        collection_name = 'test'
        data = [
            {'_id': 1, 'name': 'Item 1'},
            {'_id': 2, 'name': 'Item 2'},
            {'_id': 3, 'name': 'Item 3'}
        ]

        # Insert test data into the collection
        collection = self.dbm._db[collection_name]
        collection.insert_many(data)

        # Retrieve all test from the collection
        retrieved_data = list(self.dbm.get_all(collection_name))

        self.assertEqual(len(retrieved_data), len(data))
        for item in data:
            self.assertIn(item, retrieved_data)

    def test_get_items_with_category(self):
        # Insert test data
        collection_name = 'test'
        items = [
            {'name': 'Item 1', 'category': 'A'},
            {'name': 'Item 2', 'category': 'B'},
            {'name': 'Item 3', 'category': 'A'},
            {'name': 'Item 4', 'category': 'C'},
            {'name': 'Item 5', 'category': 'B'},
        ]
        for item in items:
            self.dbm.insert(collection_name, item)
        
        query = {'category': 'A'}  # Query for items with category 'A'
        filter = {'_id': 0}  # Exclude _id field from the results

        result = self.dbm.get(collection_name, query, filter=filter)

        self.assertIsInstance(result, pymongo.cursor.Cursor)
        self.assertEqual(len(list(result)), 2)

    def test_get_limited_items(self):
        # Insert test data
        collection_name = 'test'
        items = [
            {'name': 'Item 1', 'category': 'A'},
            {'name': 'Item 2', 'category': 'B'},
            {'name': 'Item 3', 'category': 'A'},
            {'name': 'Item 4', 'category': 'C'},
            {'name': 'Item 5', 'category': 'B'},
        ]
        for item in items:
            self.dbm.insert(collection_name, item)
        
        query = {}  # Empty query to retrieve all items
        filter = {'_id': 0}  # Exclude _id field from the results
        limit = 3  # Limit the results to 3 items

        result = self.dbm.get(collection_name, query, filter=filter, limit=limit)
        result = list(result)

        self.assertEqual(len(result), limit)  # Check the number of items returned

        self.assertNotIn('_id', result[0])  # Check that the _id field is excluded from the results

    def test_get_one_item(self):
        # Insert sample data for testing
        collection_name = 'test'
        sample_data = [
            {'name': 'Item 1', 'category': 'A'},
            {'name': 'Item 2', 'category': 'B'},
            {'name': 'Item 3', 'category': 'A'},
            {'name': 'Item 4', 'category': 'C'},
            {'name': 'Item 5', 'category': 'A'},
        ]
        self.dbm.insert_all(collection_name, sample_data)

        query = {'name': 'Item 2'}

        result = self.dbm.get_one(collection_name, query)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertEqual(result['name'], 'Item 2')

    def test_get_one_filtered_item(self):
        # Insert sample data for testing
        collection_name = 'test'
        sample_data = [
            {'name': 'Item 1', 'category': 'A'},
            {'name': 'Item 2', 'category': 'B'},
            {'name': 'Item 3', 'category': 'A'},
            {'name': 'Item 4', 'category': 'C'},
            {'name': 'Item 5', 'category': 'A'},
        ]
        self.dbm.insert_all(collection_name, sample_data)

        query = {'category': 'A'}
        filter = {'_id': 0, 'name': 1}  # Include only the name field in the result

        result = self.dbm.get_one(collection_name, query, filter)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn('name', result)
        self.assertNotIn('category', result)

        expected_names = ['Item 1', 'Item 3', 'Item 5']
        self.assertIn(result['name'], expected_names)

    def test_get_one_nonexistent_item(self):
        # Insert sample data for testing
        collection_name = 'test'
        sample_data = [
            {'name': 'Item 1', 'category': 'A'},
            {'name': 'Item 2', 'category': 'B'},
            {'name': 'Item 3', 'category': 'A'},
            {'name': 'Item 4', 'category': 'C'},
            {'name': 'Item 5', 'category': 'A'},
        ]
        self.dbm.insert_all(collection_name, sample_data)

        query = {'name': 'Nonexistent Item'}

        result = self.dbm.get_one(collection_name, query)

        self.assertIsNone(result)

    def test_update_items(self):
        # Insert sample data for testing
        collection_name = 'test'
        sample_data = [
            {'name': 'Item 1', 'category': 'A', 'quantity': 10},
            {'name': 'Item 2', 'category': 'B', 'quantity': 5},
            {'name': 'Item 3', 'category': 'A', 'quantity': 8},
            {'name': 'Item 4', 'category': 'C', 'quantity': 15},
            {'name': 'Item 5', 'category': 'A', 'quantity': 12},
        ]
        self.dbm.insert_all(collection_name, sample_data)

        query = {'category': 'A'}
        update = {'$inc': {'quantity': 2}}  # Increment quantity by 2 for matching items

        result = self.dbm.update(collection_name, query, update)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, pymongo.results.UpdateResult)
        self.assertEqual(result.matched_count, 3)  # Number of matched items
        self.assertEqual(result.modified_count, 3)  # Number of modified items

        # Verify the updated values
        updated_items = list(self.dbm.get(collection_name, query))
        self.assertEqual(updated_items[0]['quantity'], 10 + 2)
        self.assertEqual(updated_items[1]['quantity'], 8 + 2)
        self.assertEqual(updated_items[2]['quantity'], 12 + 2)

    def test_update_one_item(self):
        # Insert sample data for testing
        collection_name = 'test'
        sample_data = [
            {'name': 'Item 1', 'category': 'A', 'quantity': 10},
            {'name': 'Item 2', 'category': 'B', 'quantity': 5},
            {'name': 'Item 3', 'category': 'A', 'quantity': 8},
            {'name': 'Item 4', 'category': 'C', 'quantity': 15},
            {'name': 'Item 5', 'category': 'A', 'quantity': 12},
        ]
        self.dbm.insert_all(collection_name, sample_data)

        query = {'category': 'A'}
        update = {'$set': {'category': 'D'}}  # Change the category for the first matching item

        result = self.dbm.update_one(collection_name, query, update)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, pymongo.results.UpdateResult)
        self.assertEqual(result.matched_count, 1)  # Number of matched items
        self.assertEqual(result.modified_count, 1)  # Number of modified items

        # Verify the updated value
        updated_item = self.dbm.get_one(collection_name, {'name': 'Item 1'})
        self.assertEqual(updated_item['category'], 'D')

    def test_update_one_item_add_column(self):
        # Insert sample data for testing
        collection_name = 'test'
        sample_data = [
            {'name': 'Item 1', 'category': 'A', 'quantity': 10},
            {'name': 'Item 2', 'category': 'B', 'quantity': 5},
            {'name': 'Item 3', 'category': 'A', 'quantity': 8},
            {'name': 'Item 4', 'category': 'C', 'quantity': 15},
            {'name': 'Item 5', 'category': 'A', 'quantity': 12},
        ]
        self.dbm.insert_all(collection_name, sample_data)

        query = {'category': 'A'}
        update = {'$set': {'tag': 'test'}}  # Add the field 'tag' to the first matching item

        result = self.dbm.update_one(collection_name, query, update)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, pymongo.results.UpdateResult)
        self.assertEqual(result.matched_count, 1)  # Number of matched items
        self.assertEqual(result.modified_count, 1)  # Number of modified items

        # Verify the updated value
        updated_item = self.dbm.get_one(collection_name, query)
        self.assertEqual(updated_item['tag'], 'test')

    def test_update_one_nonexistent_item(self):
        # Insert sample data for testing
        collection_name = 'test'
        sample_data = [
            {'name': 'Item 1', 'category': 'A', 'quantity': 10},
            {'name': 'Item 2', 'category': 'B', 'quantity': 5},
            {'name': 'Item 3', 'category': 'A', 'quantity': 8},
            {'name': 'Item 4', 'category': 'C', 'quantity': 15},
            {'name': 'Item 5', 'category': 'A', 'quantity': 12},
        ]
        self.dbm.insert_all(collection_name, sample_data)

        query = {'name': 'Nonexistent Item'}
        update = {'$set': {'category': 'E'}}  # Update the category for a non-existent item

        result = self.dbm.update_one(collection_name, query, update)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, pymongo.results.UpdateResult)
        self.assertEqual(result.matched_count, 0)  # No matched items
        self.assertEqual(result.modified_count, 0)  # No modified items

    def test_export_collection(self):
        """Test exporting a collection to a JSON file."""
        collection_name = 'test'
        data = [
            {'_id': 1, 'name': 'Item 1'},
            {'_id': 2, 'name': 'Item 2'},
            {'_id': 3, 'name': 'Item 3'}
        ]

        # Insert test data into the collection
        collection = self.dbm._db[collection_name]
        collection.insert_many(data)

        # Mock the tqdm module to suppress progress bar output
        tqdm_mock = MagicMock()
        tqdm_mock.return_value = data

        # Mock the open function to capture the exported file contents
        open_mock = MagicMock()
        open_mock.return_value.__enter__.return_value.write = MagicMock()

        with unittest.mock.patch('GPRS.database.tqdm', tqdm_mock), \
                unittest.mock.patch('GPRS.database.open', open_mock):
            self.dbm.export_collection(os.path.join('path', 'to', 'export'), collection_name)

        # Ensure that the exported file is written with the correct contents
        open_mock.assert_called_once_with(os.path.join('path', 'to', 'export', 'test.json'), 'w')
        write_mock = open_mock.return_value.__enter__.return_value.write
        #write_mock.assert_called_once_with('[{"_id": 1, "name": "Item 1"},{"_id": 2, "name": "Item 2"},{"_id": 3, "name": "Item 3"}]')
        write_mock.assert_has_calls([unittest.mock.call(line) for line in ['[', '{"_id": 1, "name": "Item 1"}', ',', '{"_id": 2, "name": "Item 2"}', ',', '{"_id": 3, "name": "Item 3"}', ']']])

if __name__ == '__main__':
    unittest.main
