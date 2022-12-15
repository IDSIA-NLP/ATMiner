import bconv
from bconv.doc.document import RelationMember

# doc = bconv.load('../data/tmp/input/PMC3082961.txt', fmt='txt')
doc = bconv.load('../data/tmp/input/test_BC5CDR.json', fmt='bioc_json')

# print(doc[0].text)
print(doc)
print(doc[0])

for section in doc:
    print(section)
    for sentence in section:
        print(sentence)


print(doc[0][1].text)

# Entity(id, text, spans, meta)
ent = bconv.Entity(None, "Intravenous administration", [(0,26)], {"type": "some_type"}) 
doc[0][1].add_entities([ent])

sent_ents = [ [e.id, e.start, e.end, e.text, e.metadata ] for e in list(doc[0][1].iter_entities())]

print(sent_ents)


print(doc.text)

print(doc.text[36:62])

doc_ents = [ [e.id, e.start, e.end, e.text, e.metadata ] for e in list(doc.iter_entities())]

print(doc_ents)
print(doc[0].relations)

for r in doc[0].iter_relations():
    print([r.metadata, r.id])
    for r_member in r:
        print(r_member)

# (("refid", "role"),...,("refid", "role"))
rel_membmers = (("6", "role_A"),("5", "role_B"))
new_rel = bconv.Relation("122", rel_membmers)
new_rel.metadata = {"type": "hasTestRelation"}

doc[0].relations.append(new_rel)

print("\nBefore sanitization relations")
for r in doc[0].iter_relations():
    print([r.metadata, r.id])
    for r_member in r:
        print(r_member)

# Check reference IDs in relations.
doc[0].sanitize_relations()

print("\nAfter sanitization relations")
for r in doc[0].iter_relations():
    print([r.metadata, r.id])
    for r_member in r:
        print(r_member)


print("Document entities:")
print("Document entities len:", len(list(doc[0].iter_entities())))