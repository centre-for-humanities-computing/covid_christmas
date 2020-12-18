import pandas as pd

import utils.stanza_to_df as std
import utils.flair_ner as ner
import utils.twitter_utils as tu

df = pd.read_csv("tweet-jul-dk-novdec-2020.csv", header=None)
df.columns = ["id", "date", "text"]

# Tokenize, Lemma, POS, dependency
dfs = std.stanza_to_df(df.text,
                       langs="da",
                       verbose=True)

# NER
dfs = ner.flair_tagger_ttt(dfs, langs="da")

# Twitter
dfs = tu.twitter_format(dfs)
dfs = list(dfs)

df["preprocessed"] = dfs
df.to_pickle("preprocessed.pkl")
# dfer = pd.read_pickle("preprocessed.pkl")
# dfer["preprocessed"][0] # test