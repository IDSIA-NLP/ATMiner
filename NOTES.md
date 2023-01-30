# Notes

# Entity Recognition

*NEEDS TO BE UPDATED*

The `EntityRecognizer` should take a text string and retuern the a list of dicts with:
```python
[{
    "id_" : "..."
    "text" : "..."
    "start" : "..."
    "end" : "..."
    "ent_type" : "..."
    "preferred_form" : "..."
    "resource" : "..."
    "native_id" : "..."
    "cui" : "..."
    ... + extra_info 
}]
```