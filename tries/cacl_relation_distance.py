import bconv
import math
from glob import glob
import pandas as pd
annotator = ["rmwaterhouse", "MorganeMassy"]
stats = []
words = []
for anno in annotator:

    files = glob(f'../data/tmp/input_eval/{anno}/*.bioc.json')

    for file in files:

        coll = bconv.load(file, fmt='bioc_json', byte_offsets=False)

        doc_words = coll[0].text.split()
        
        sent_words_len = []
        for section in coll[0]:
            for sentence in section:
                sent_words_len += [int(len(sentence.text) / 3.02)]

        
        words += doc_words

        print(words[:10])
        ents = {e.id: {"start":e.start, "end": e.end} for e in list(coll[0].iter_entities())}

        for r in coll[0].iter_relations():
            # print([r.metadata, r.id])
            members = [(ents[r[0].refid]["start"], ents[r[0].refid]["end"]), (ents[r[1].refid]["start"], ents[r[1].refid]["end"])]
            members = sorted(members)
            # print(members)
            stats.append({"annotator":anno, "doc_id": coll[0].id, "id": r.id, "type": r.metadata["type"], "distance": members[1][0] - members[0][1]})

# print(stats)
df = pd.DataFrame(stats)
df.to_csv("./eval_stats.tsv", index=False, sep="\t")

# plot of distance between entities in relation by type of relation
ax = df[["distance","annotator"]].plot.hist(bins=12, alpha=0.5, by="annotator", figsize=(10, 10))

print("average words length: ", sum([len(w) for w in words])/len(words))
# save plot
ax.figure.savefig("./eval_stats.png")


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

tok_words = []
tok_words += [tokenizer.tokenize(w) for w in words]
tok_words = [w for l in tok_words for w in l]
tok_words = [w.replace("#","") for w in tok_words]

print("average tok_words length: ", sum([len(w) for w in tok_words])/len(tok_words))
print("average sent_words_len: ", sum(sent_words_len)/len(sent_words_len))