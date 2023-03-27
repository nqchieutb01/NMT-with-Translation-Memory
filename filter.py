import tantivy
import os
from tqdm import tqdm
# Declaring our schema.
import re
import string
PUNC = string.punctuation
schema_builder = tantivy.SchemaBuilder()
schema_builder.add_text_field("title", stored=True)
schema = schema_builder.build()

# Creating our index (in memory, but filesystem is available too)
index = tantivy.Index(schema, path=os.getcwd() + '/index1')
# # Adding one document.
# writer = index.writer()
# lines = []
# with open("data/train_rmd.vi","r") as f:
#     for line in f:
#         lines.append(line.strip())
# for line in lines:
#     writer.add_document(tantivy.Document(
#         title=[line],
#     ))
# # ... and committing
# writer.commit()

index.reload()
TOPK = 5
searcher = index.searcher()

test = []
with open("data/train_rmd.vi", 'r') as f:
    for line in f:
        test.append(line.strip())


with open("data/train_rmd.vi.tmp", 'w') as f:
    for i in tqdm(range(len(test))):
        # print(test[i])
        # test[i] = "He passed away , history of fallen over "
        # query = index.parse_query(test[i], ["title"])
        # arr = searcher.search(query, TOPK + 1).hits
        # print(arr)
        # exit(0)
        test[i] = test[i].strip()
        _ = test[i].translate(str.maketrans(PUNC, ' ' * len(PUNC)))

        query = index.parse_query(_, ["title"])
        tmp = ''
        arr = searcher.search(query, TOPK + 1).hits
        if len(arr)==0:
            # print('ok')
            print(_)
            for j in range(TOPK):
                tmp += '\t' + _ + '\t' + str(1.0)
            f.write(tmp + '\n')
            continue
        max_score = arr[0][0] * 1.1
        cnt = 0
        for j in range(len(arr)):
            best_doc = searcher.doc(arr[j][1])
            if best_doc['title'][0] != test[i] and cnt < TOPK:
            # if cnt < TOPK:
                cnt += 1
                tmp += '\t' + best_doc['title'][0] + '\t' + str(arr[j][0] / max_score)
            # print(best_doc['title'][0])
            # print(arr[i][0])
        f.write(tmp + '\n')
        # break
