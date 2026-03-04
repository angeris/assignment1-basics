# %%
from cs336_basics.tokenizer import Tokenizer
import pickle

with open("output/bpe_tinystories.pkl", "rb") as f:
    tinystories_bpe = pickle.load(f)

tinystories_special_tokens = ["<|endoftext|>"]

tinystories_tokenizer = Tokenizer(tinystories_bpe['vocab'], tinystories_bpe['merges'], tinystories_special_tokens)
