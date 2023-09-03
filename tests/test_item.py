"""
test_item.py

This file contains the unit tests for the Item class.

Author: Gustav Hubert
"""

import unittest
from GPRS.item import Item

import unittest
from GPRS.item import Item

class ItemTest(unittest.TestCase):
    """Unit tests for the Item class."""

    def setUp(self):
        """Set up a sample Item instance for testing."""
        self.item = Item("Sample Item", "https://example.com/item", ["tag1", "tag2"])

    def test_add_tag(self):
        """Test the add_tag method of Item."""
        self.item.add_tag("tag3Name", "tag3Value")
        self.assertIn("tag3Name", self.item.get_tags())

    def test_remove_tag(self):
        """Test the remove_tag method of Item."""
        self.item.remove_tag("tag1")
        self.assertNotIn("tag1", self.item.get_tags())

    def test_update_title(self):
        """Test the update_title method of Item."""
        new_title = "Updated Item Title"
        self.item.update_title(new_title)
        self.assertEqual(self.item.title, new_title)

    def test_update_source(self):
        """Test the update_source method of Item."""
        new_source = "https://example.com/updated-item"
        self.item.update_source(new_source)
        self.assertEqual(self.item.source, new_source)

    def test_get_tags(self):
        """Test the get_tags method of Item."""
        tags = self.item.get_tags()
        self.assertIsInstance(tags, list)
        self.assertCountEqual(tags, ["tag1", "tag2"])

if __name__ == '__main__':
    unittest.main()