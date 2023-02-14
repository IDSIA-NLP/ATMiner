import bconv
from bconv.doc.document import RelationMember

doc = bconv.load('../data/tmp/input/test_BC5CDR.json', fmt='bioc_json')
#doc = bconv.load('../data/tmp/input/PMC3082961.txt', fmt='txt')

# # print(doc[0].text)
# print(type(doc))
# print(type(doc) == bconv.doc.document.Collection)
# print(type(doc) == bconv.doc.document.Document)
# print(doc.id)
# print(doc.filename)

# print(doc)
# print(doc[0])

# for section in doc:
#     print(section)
#     for sentence in section:
#         print(sentence)
#         # ent = bconv.Entity(None, sentence.text[0:1], [(0,1)], {"type": "some_type"}) 
#         # sentence.add_entities([ent])




# print(doc[0][1].text)

# #Entity(id, text, spans, meta)
# ent = bconv.Entity(None, "Intravenous administration", [(0,26)], {"type": "some_type"}) 
# doc[0][1].add_entities([ent])

# sent_ents = [ [e.id, e.start, e.end, e.text, e.metadata ] for e in list(doc[0][1].iter_entities())]

# print(sent_ents)


# print(doc.text)

# print(doc.text[36:62])

# doc_ents = [ [e.id, e.start, e.end, e.text, e.metadata ] for e in list(doc.iter_entities())]

# print(doc_ents)
# print(doc[0].relations)

# for r in doc[0].iter_relations():
#     print([r.metadata, r.id])
#     for r_member in r:
#         print(r_member)

# # (("refid", "role"),...,("refid", "role"))
# rel_membmers = (("6", "role_A"),("5", "role_B"))
# new_rel = bconv.Relation("122", rel_membmers)
# new_rel.metadata = {"type": "hasTestRelation"}

# doc[0].relations.append(new_rel)

# print("\nBefore sanitization relations")
# for r in doc[0].iter_relations():
#     print([r.metadata, r.id])
#     for r_member in r:
#         print(r_member)

# # Check reference IDs in relations.
# doc[0].sanitize_relations()

# print("\nAfter sanitization relations")
# for r in doc[0].iter_relations():
#     print([r.metadata, r.id])
#     for r_member in r:
#         print(r_member)


# print("Document entities:")
# print("Document entities len:", len(list(doc[0].iter_entities())))
# # print("Document entities len:", len(list(doc.iter_entities())))
# # print("Document entities len:", [ [e.id, e.start, e.end, e.text, e.metadata ] for e in list(doc.iter_entities())])

# with open('../data/tmp/output/test_output_BC5CDR.json', 'w', encoding='utf8') as f:
#     bconv.dump(doc, f, fmt='bioc_json')



#! Tests for 
coll = bconv.load('../data/tmp/input/awqipQxjojnzAwukvMo3ZZv9r3KW-zookeys.687.13164.bioc.json', fmt='bioc_json', byte_offsets=False)
#doc = bconv.load('../data/tmp/input/PMC3082961.txt', fmt='txt')

# print(doc[0].text)
print(type(coll))
print(type(coll) == bconv.doc.document.Collection)
print(type(coll) == bconv.doc.document.Document)

print(coll)

# prev_end = 0
# for doc in coll:
#     for section in doc:
#         print("Section:", section)
#         for sentence in section:
#             print("Sentence: ", sentence)
#             # ent = bconv.Entity(None, sentence.text[0:1], [(0,1)], {"type": "some_type"}) 
#             # sentence.add_entities([ent])

#             sent_ents = [ [e.id, e.start, e.end, e.text, e.metadata ] for e in list(sentence.iter_entities())]

#             #print("Sent vars: ", vars(sentence))
#             print("Sent start offset: ", sentence._start)
#             print("Sent end offset: ", sentence._end)

#             if prev_end == sentence._start:
#                 print("GLEICH")
#             else:
#                 print("\n\nNOT-EQUAL\n\n")

#             prev_end = sentence._end
#             print("Sent text: ", sentence.text)
#             print("Sent entities: ", sent_ents)
#             # print()
#             # if sent_ents:
#             #     print(sentence.text[sent_ents[0][1]-sentence._start:sent_ents[0][2]-sentence._start] == sent_ents[0][3])
#             #     print(sentence.text[sent_ents[0][1]-sentence._start:sent_ents[0][2]-sentence._start], sent_ents[0][3])

for doc in coll:
    print(f"Doc length: {len(list(doc.iter_entities()))}")

    # --------- Entities --------------
    # Delete all entities
    for pa in doc:
        for sent in pa:
            sent.entities = []

    print(f"Doc length: {len(list(doc.iter_entities()))}")

    # Add new entities
    for pa in doc:
        for idx, sent in enumerate(pa[:3]):
            ent = bconv.Entity(idx, sent.text[0:1], [(0,1)], {"type": "some_type"}) 
            sent.add_entities([ent])
            print("Added entity")

    
    # --------- Relations --------------
    print(f"Num rel before: {len(list(doc.iter_relations()))}")
    doc.relations = []
    print(f"Num rel after: {len(list(doc.iter_relations()))}")

    node_1 = ((1, "some_type"),(2, "some_type"))
    new_rel_1 = bconv.Relation(1, node_1)
    new_rel_1.metadata = {"type": "fake_type",
            "context_start_char": 0,
            "context_end_char": -1,
            "context_size": 888,
            "annotator": "fake_anno"}


    node_2 = ((2, "some_type"),(3, "some_type"))
    new_rel_2 = bconv.Relation(2, node_2)
    new_rel_2.metadata = {"type": "fake_type",
            "context_start_char": 0,
            "context_end_char": -1,
            "context_size": 888,
            "annotator": "fake_anno"}

    doc.relations.append(new_rel_1)
    doc.relations.append(new_rel_2)

    print(f"Num entities: {len(list(doc.iter_entities()))}")
    print(f"Num relations: {len(list(doc.iter_relations()))}")


    for e in list(doc.iter_entities()):
        print(f"{e.id}, {e.start}, {e.end}, {e.text}, {e.metadata} ")
        
    for r in list(doc.iter_relations()):
        print(f"{r.id}, {r.type}, {r._children}, {r._children[0].refid}, {r.metadata}")


    print(f"Num entities: {len(list(doc.iter_entities()))}")
    print(f"Num relations: {len(list(doc.iter_relations()))}")