#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description: ...
Author: ...
Month Year
"""
import uuid

class ATEntity(object):
    """ A class to represent an AT Entity.

    Args:
        text (str): The text of the entity.
        spans (list): A list of tuples containing the start and end character offsets of the entity.
        ent_type (str): The type of the entity e.g. "Arthropod".
        preferred_form (str, optional): The preferred form of the entity. Defaults to "".
        resource (str, optional): The resource the entity was extracted from. Defaults to "".
        native_id (str, optional): The native id of the entity. Defaults to "".
        cui (str, optional): The CUI of the entity. Defaults to "".
        extra_info (dict, optional): A dictionary containing extra information about the entity. Defaults to None.
    """
    def __init__(self, text, spans, ent_type, preferred_form="", resource="", native_id="", cui="", extra_info=None):
        """Initialize the ATEntity object."""
        self.id_ = uuid.uuid4().hex
        self.text = text
        self.spans = sorted((start, end) for start, end in spans)
        
        self.metadata = {}
        self.metadata["type"] = ent_type
        self.metadata["preferred_form"]  = preferred_form
        self.metadata["resource"]  = resource
        self.metadata["native_id"]  = native_id
        self.metadata["cui"]  = cui

        if type(extra_info) == dict or extra_info == None:
            if not extra_info == None:
                if not set(extra_info.keys()) & set(self.metadata.keys()):
                    self.metadata.update(extra_info)
                else:
                    raise ValueError("Extra-info cannot have overlapping keys with metadata.")
        else:
            raise ValueError("Extra-info must be type of dict or None.")

    @property
    def ent_type(self):
        return self.metadata["type"]

    @ent_type.setter
    def ent_type(self, ent_type):
        self.metadata["type"] = ent_type

    @property
    def preferred_form(self):
        return self.metadata["preferred_form"]

    @preferred_form.setter
    def preferred_form(self, preferred_form):
        self.metadata["preferred_form"] = preferred_form

    @property
    def resource(self):
        return self.metadata["resource"]

    @resource.setter
    def resource(self, resource):
        self.metadata["resource"] = resource
    
    @property
    def native_id(self):
        return self.metadata["native_id"]

    @native_id.setter
    def native_id(self, native_id):
        self.metadata["native_id"] = native_id

    @property
    def cui(self):
        return self.metadata["cui"]

    @cui.setter
    def cui(self, cui):
        self.metadata["cui"] = cui


    def shift_offset(self, shift_by):
        """Shift the offsets of the entity by a given number of characters.
        
        Args:
            shift_by (int): The number of characters to shift the offsets by.
        """
        
        self.start += shift_by
        self.end += shift_by


    def update_metadata(self, extra_info):
        """Update the metadata of the entity with extra information.
        
        Args:
            extra_info (dict): A dictionary containing extra information about the entity.
        """

        if type(extra_info) == dict:
            if not set(extra_info.keys()) & set(self.metadata.keys()):
                self.metadata.update(extra_info)
            else:
                raise ValueError("Extra-info cannot have overlapping keys with metadata.")
        else:
            raise ValueError("Extra-info must be type of dict")

    
