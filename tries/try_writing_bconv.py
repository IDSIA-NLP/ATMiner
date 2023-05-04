import bconv

coll = bconv.load('../data/tmp/input/awqipQxjojnzAwukvMo3ZZv9r3KW-zookeys.687.13164.bioc.json', fmt='bioc_json', byte_offsets=False)


with open("./test_byte-offset-true.bioc.json", 'w', encoding='utf8') as f:
    #TODO: Might need more specification of the different output formats options
    bconv.dump(coll, f, fmt="bioc_json", byte_offsets=True)