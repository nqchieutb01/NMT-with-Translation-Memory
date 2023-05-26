import argparse
from tqdm import tqdm
import sys
import os
# sys.path.append("/home/chieunq/Documents/code/MT-Km-Vi/lib/deep-translator")
from deep_translator import GoogleTranslator

my_parser = argparse.ArgumentParser(description='List the content of a folder')
my_parser.add_argument('id_chunk',
                       metavar='id_chunk',
                       type=int,
                       help='the path to list')
args = my_parser.parse_args()

id_chunk = args.id_chunk
input_path = "v" + str(id_chunk)
batch_sz = 16

STORE_DATA = "anhdat"
DATA_VI = "/home/chieunq/Downloads/utterances.csv"

lefts =  [0,5e4, 1e5, 1e5+5e4, 2e5, 2e5+5e4,3e5]
rights = [5e4,1e5, 1e5+5e4, 2e5, 2e5+5e4, 3e5,4e5]
# if not os.path.exists(f"{STORE_DATA}{input_path}"):
#     os.makedirs(f"{STORE_DATA}{input_path}")


def start():
    proxies_example = {
        "https": "10.30.153.169:3128",  # example: 34.195.196.27:8080
        "http": "10.30.153.169:3128"
    }
    model = GoogleTranslator(source='en', target='vi', proxies=proxies_example)

    src_vi = []
    index = 0

    with open(f"{DATA_VI}") as f2:
        for line in f2:
            if rights[id_chunk] > index >= lefts[id_chunk]:
                src_vi.append(line.strip())
            index += 1

    print(len(src_vi))

    with open(f'{STORE_DATA}{input_path}_trans.txt', 'a') as f:
        sens = ""
        for i in tqdm(range(0, len(src_vi), batch_sz)):
            left = i
            right = min(len(src_vi), i + batch_sz)
            ori = src_vi[left:right]
            sens = '\n'.join(ori)
            try:
                tmp = model.translate(sens)
            except Exception as e:
                print(e)
                continue
            tmp = tmp.split('\n')
            for j in range(right-left):
                f.write(tmp[j] + '\t' + ori[j] + '\n')
start()
