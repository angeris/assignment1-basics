# %%
import pickle

with open("./output/bpe_owt.pkl", "rb") as f:
    s = pickle.load(f)
    merges = s['merges']


with open("./output/bpe_owt_valid.pkl", "rb") as f:
    sp = pickle.load(f)
    mergesp = sp['merges']
