import bconv
from bconv.doc.document import RelationMember
from io import StringIO

doc = bconv.load('../data/tmp/input/test_BC5CDR.json', fmt='bioc_json')

fp = StringIO()


bconv.dump(doc, fp, fmt='conll', tagset='IOB', include_offsets=True)

print(fp.getvalue())


USE_IOB2_FORMAT = True

def conll_to_list(lines):
    out_list = []
    d = {
        'tokens':[],
        'tags':[]
    }
    for line in lines:
        
        if not line.startswith("#"):
            if not line.isspace():
                e = line.strip().split('\t')
                print(e)
                d["tokens"].append(e[0]) 
                if USE_IOB2_FORMAT:
                    d["tags"].append(e[-1].replace("S-","B-").replace("E-","I-"))
                else:
                    d["tags"].append(e[-1])
            else:
                if d["tokens"] and d["tags"]:
                    if d['tokens']:
                        out_list.append(d)
                        d = {
                            'tokens':[],
                            'tags':[]
                        }

    return out_list

lines = fp.getvalue().splitlines(True)
print(lines)

token_tags_list = conll_to_list(lines)
print(token_tags_list)

# Check if token split is the same

tag_list_true = [e["tags"] for e in token_tags_list]
test_list_pred = [['O'] * len(e["tags"]) for e in token_tags_list]

print(tag_list_true)




cls_report_strict = classification_report(tag_list_true, test_list_pred, mode="strict" , scheme=IOB2)
print("STRICT:\n", cls_report_strict)

cls_report_conlleval = classification_report(tag_list_true, test_list_pred, mode=None , scheme=IOB2)
print("CONLLEVAL equivalent:\n", cls_report_conlleval)
