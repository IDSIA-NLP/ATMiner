
gold_relations_with_entities = [
    {
    "id":1, 
    "type":"hasTrait", 
    "metadata": {}, 
    "entities": [
                {"entity": {"id":0, "start": 1, "end": 2, "text":"a", "type": "Arthropod"}, "role":"Arthropod"},
                {"entity": {"id":1, "start": 3, "end": 4, "text":"b", "type": "Trait"}, "role":"Trait"},
    ]},
    {
    "id":2, 
    "type":"hasTrait", 
    "metadata": {}, 
    "entities": [
                {"entity": {"id":3, "start": 7, "end": 8, "text":"a", "type": "Arthropod"}, "role":"Arthropod"},
                {"entity": {"id":2, "start": 5, "end": 6, "text":"b", "type": "Trait"}, "role":"Trait"},
    ]},
    {
    "id":3, 
    "type":"hasTrait", 
    "metadata": {}, 
    "entities": [
                {"entity": {"id":4, "start": 9, "end": 10, "text":"a", "type": "Arthropod"}, "role":"Arthropod"},
                {"entity": {"id":5, "start": 11, "end": 12, "text":"b", "type": "Trait"}, "role":"Trait"},
    ]},
    {
    "id":4, 
    "type":"hasTrait", 
    "metadata": {}, 
    "entities": [
                {"entity": {"id":6, "start": 13, "end": 14, "text":"a", "type": "Arthropod"}, "role":"Arthropod"},
                {"entity": {"id":7, "start": 15, "end": 16, "text":"b", "type": "Trait"}, "role":"Trait"},
    ]},
    {
    "id":5, 
    "type":"hasTrait", 
    "metadata": {}, 
    "entities": [
                {"entity": {"id":8, "start": 17, "end": 18, "text":"a", "type": "Arthropod"}, "role":"Arthropod"},
                {"entity": {"id":9, "start": 19, "end": 20, "text":"b", "type": "Trait"}, "role":"Trait"},
    ]},
]

pred_relations_with_gold_entities = [
    {
    "id":11, 
    "type":"hasTrait", 
    "metadata": {}, 
    "entities": [
                {"entity": {"id":0, "start": 1, "end": 2, "text":"a", "type": "Arthropod"}, "role":"Arthropod"},
                {"entity": {"id":11, "start": 3, "end": 4, "text":"b", "type": "Trait"}, "role":"Trait"},
    ]},
    # Change entities order
    {
    "id":22, 
    "type":"hasTrait", 
    "metadata": {}, 
    "entities": [
                {"entity": {"id":22, "start": 5, "end": 6, "text":"b", "type": "Trait"}, "role":"Trait"},
                {"entity": {"id":33, "start": 7, "end": 8, "text":"a", "type": "Arthropod"}, "role":"Arthropod"},
                
    ]},
    
    # Change label annotation
    {
    "id":33, 
    "type":"none", 
    "metadata": {}, 
    "entities": [
                {"entity": {"id":44, "start": 9, "end": 10, "text":"a", "type": "Arthropod"}, "role":"Arthropod"},
                {"entity": {"id":55, "start": 11, "end": 12, "text":"b", "type": "Trait"}, "role":"Trait"},
    ]},
    
    # Switch entity annotation
    {
    "id":44, 
    "type":"hasTrait", 
    "metadata": {}, 
    "entities": [
                {"entity": {"id":66, "start": 13, "end": 14, "text":"a", "type": "Trait"}, "role":"Trait"},
                {"entity": {"id":77, "start": 15, "end": 16, "text":"b", "type": "Arthropod"}, "role":"Arthropod"},
    ]},
    
    # Change label and entity type
    {
    "id":55, 
    "type":"hasValue", 
    "metadata": {}, 
    "entities": [
                {"entity": {"id":88, "start": 17, "end": 18, "text":"a", "type": "Value"}, "role":"Value"},
                {"entity": {"id":99, "start": 19, "end": 20, "text":"b", "type": "Trait"}, "role":"Trait"},
    ]},
    
]

gold_labels = []
predicted_labels = []
for gold_rel in gold_relations_with_entities:
    gold_rel["entities"] = sorted(gold_rel["entities"], key=lambda d: d["entity"]['start'])
    print("GOLD:", gold_rel)
    for pred_rel in pred_relations_with_gold_entities:
        print("PRED:", pred_rel)
        pred_rel["entities"] = sorted(pred_rel["entities"], key=lambda d: d["entity"]['start'])
        
        if gold_rel["entities"][0]["entity"]["start"] == pred_rel["entities"][0]["entity"]["start"] \
        and gold_rel["entities"][0]["entity"]["end"] == pred_rel["entities"][0]["entity"]["end"] \
        and gold_rel["entities"][1]["entity"]["start"] == pred_rel["entities"][1]["entity"]["start"] \
        and gold_rel["entities"][1]["entity"]["end"] == pred_rel["entities"][1]["entity"]["end"]:
            
            print(f"\nMATCH IDS gold: {gold_rel['id']},pred: {pred_rel['id']}\n")
            if gold_rel["type"] == pred_rel["type"] \
            and (gold_rel["entities"][0]["role"] != pred_rel["entities"][0]["role"] \
                or gold_rel["entities"][1]["role"] != pred_rel["entities"][1]["role"]):
                # Same relation label but the entities are labeled in the opposite way
                # --> force the relation labels to disagree
                gold_labels.append(gold_rel["type"])
                predicted_labels.append("none")
                break
            else:
                gold_labels.append(gold_rel["type"])
                predicted_labels.append(pred_rel["type"])
                break
        
        print()
from sklearn.metrics import classification_report
#y_true = ["a", "b", "c", "a"]
#y_pred = ["a", "b", "c", "c"]
print("gold_labels", gold_labels)
print("predicted_labels", predicted_labels)



print(classification_report(gold_labels, predicted_labels))

print(classification_report(gold_labels, predicted_labels, zero_division=1))

print(classification_report(gold_labels, predicted_labels, labels=["hasValue", "hasTrait"]))

print(classification_report(gold_labels, predicted_labels, labels=["hasValue", "hasTrait"], zero_division=1))