import re
dup = {}
vis,ens = [],[]

with open("data/train.en","r") as f1, open("data/train.vi","r") as f2:
    for (km,vi) in zip(f1,f2):
        km = re.sub(' +', ' ', km.strip())
        vi = re.sub(' +', ' ', vi.strip())
        km = re.sub('\u200b', ' ', km)
        vi = re.sub('\u200b', ' ', vi)
        km = ' '.join(km.split())
        vi = ' '.join(vi.split())
        if km not in dup and vi not in dup:
            dup[km] = 1
            dup[vi] = 1
            ens.append(km)
            vis.append(vi)

with open("data/train_rmd.en","w") as f1, open("data/train_rmd.vi","w") as f2:
    for i in range(len(vis)):
        f1.write(ens[i] +'\n')
        f2.write(vis[i] + '\n')
