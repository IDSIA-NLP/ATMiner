#!/usr/bin/env python3
# coding: utf8


'''
Wrapper for renaming a built-in postfilter.
'''


from oger.post import remove_sametype_submatches as postfilter

# The postfilter is stored using the function's __name__ property.
postfilter.__name__ = 'longest_match'
