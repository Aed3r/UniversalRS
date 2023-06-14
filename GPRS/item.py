"""
item.py

This module defines the `Item` class, which represents an item with a title, source, and tags. Each item is assigned a unique ID upon creation. The class provides methods to add and remove tags, update the title and source, and retrieve the tags associated with the item. The class also implements a string representation for easy printing.

Author: Gustav Hubert
"""

import uuid, unittest

class Item:
    """Represents an item with a title, source, and tags."""

    def __init__(self, title, source, type, tags=None):
        """
        Initialize a new Item instance.

        Args:
            title (str): The title of the item.
            source (str): The source of the item.
            tags (list, optional): The tags associated with the item. Defaults to None.
        """
        self.id = str(uuid.uuid4())
        self.title = title
        self.source = source
        self.type = type
        self.tags = tags if tags is not None else {}

    def add_tag(self, tagName, tagValue):
        """
        Add a new tag to the item.

        Args:
            tag (str): The tag to add.
        """
        self.tags[tagName] = tagValue

    def remove_tag(self, tag):
        """
        Remove a tag from the item.

        Args:
            tag (str): The tag to remove.
        """
        if tag in self.tags:
            self.tags.remove(tag)

    def update_title(self, new_title):
        """
        Update the title of the item.

        Args:
            new_title (str): The new title for the item.
        """
        self.title = new_title

    def update_source(self, new_source):
        """
        Update the source of the item.

        Args:
            new_source (str): The new source for the item.
        """
        self.source = new_source

    def update_type(self, new_type):
        """
        Update the type of the item.

        Args:
            new_type (str): The new type for the item.
        """
        self.type = new_type

    def get_id(self):
        """
        Get the ID of the item.

        Returns:
            str: The ID of the item.
        """
        return self.id
    
    def get_title(self):
        """
        Get the title of the item.

        Returns:
            str: The title of the item.
        """
        return self.title

    def get_tags(self):
        """
        Get the tags associated with the item.

        Returns:
            list: The list of tags.
        """
        return self.tags
    
    def get_type(self):
        """
        Get the type of the item.

        Returns:
            str: The type of the item.
        """
        return self.type

    def __str__(self):
        """
        Get a string representation of the item.

        Returns:
            str: The string representation of the item.
        """
        return f"ID: {self.id}\nTitle: {self.title}\nSource: {self.source}\nType: {self.type}\nTags: {', '.join(self.tags)}"
