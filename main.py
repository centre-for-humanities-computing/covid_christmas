from spacy.lang.da.stop_words import STOP_WORDS
import pandas as pd
import gensim
import pyLDAvis.gensim
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np

from collections import Counter

import utils.stanza_to_df as std
import utils.flair_ner as ner
import utils.twitter_utils as tu
from utils.wordcloud import tokenlist_wordcloud
from vaderSentiment.vaderSentiment_da import (
    SentimentIntensityAnalyzer as Sentiment_da,
)
df = pd.read_pickle("preprocessed.pkl")
len(df)
dfs = df["preprocessed"]

len(pd.read_csv("tweet-jul-dk-novdec-2020.csv", header=None))


def topic_filter(df, pos_to_keep=["NOUN", "ADJ", "VERB", "PRON", "SYM_EMOJI"],
                 keep_ner=True):
    if keep_ner:
        res = df[(df["upos"].isin(pos_to_keep)) |
                 ~(df["ner"].isin(["O", "URL"]))]
    else:
        res = df[(df["upos"].isin(pos_to_keep))]
    return res["lemma"].tolist()


# create wordclouds
bow = [token for df in dfs for token in topic_filter(df)
       if token not in STOP_WORDS]  # flat
tokenlist_wordcloud(bow, mask_img="wordcloud_masks/ChristmasTree_mask.jpg",
                    save_as="wordcloud_img/christmas_tree.png")
tokenlist_wordcloud(bow, mask_img="wordcloud_masks/nissehue.png",
                    save_as="wordcloud_img/nissehue.png")
tokenlist_wordcloud(bow, mask_img="wordcloud_masks/raindeer_mask2.png",
                    save_as="wordcloud_img/raindeer.png")

# topic model
filtered = [topic_filter(df) for df in dfs]
filtered = [t for t in filtered if len(t) > 10]
filtered = [[token for token in text if token not in STOP_WORDS]
            for text in filtered]
dictionary = gensim.corpora.dictionary.Dictionary(filtered)
dictionary.filter_extremes(no_below=5, no_above=0.5)
bow_corpus = [dictionary.doc2bow(toks) for toks in filtered]

lda = gensim.models.ldamodel.LdaModel(bow_corpus, num_topics=15)
pyLDAvis.enable_notebook()
p = pyLDAvis.gensim.prepare(lda, bow_corpus, dictionary)
pyLDAvis.save_html(p, 'lda_christmas.html')
p

# most popular  hashtag
sns.set_theme(style="darkgrid")
x = [i for df in dfs for i in df["lemma"]
     [df["ner"] == "HASHTAG"]]
x = [x for x in x if x.startswith("#")]
x = [i for i in x if i in {i[0] for i in Counter(x).most_common(10)}]

plt.xticks(rotation=45)
ax = sns.countplot(x=x)
ax.set_title('Most Popular Hashtag on Danish Christmas Twitter')
ax.set_ylabel('Count')
plt.savefig('plots/hashtag.png', bbox_inches="tight")

# most popular mentions
x = [i for df in dfs for i in df["lemma"]
     [df["ner"] == "TWITTER_USER"]]
x = [x for x in x if x.startswith("@")]
x = [i for i in x if i in {i[0] for i in Counter(x).most_common(10)}]

plt.xticks(rotation=45)
ax = sns.countplot(x=x)
ax.set_title('Most Popular Mention on Danish Christmas Twitter')
ax.set_ylabel('Count')
plt.savefig('plots/mentions.png', bbox_inches="tight")

# sentiment time series
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['date_hours'] = df['date'].dt.round("H")
df['date_days'] = df['date'].dt.round("D")

df["lemma"] = [d["lemma"].tolist() for d in dfs]


analyser = Sentiment_da()
sentiment = [analyser.polarity_scores(text, tokenlist=lemma)
             for text, lemma in zip(df["text"], df["lemma"])]
df["neg"], df["pos"], df["neu"], df["compound"] = zip(
    *[(s["neg"], s["pos"], s["neu"], s["compound"]) for s in sentiment])


plt.xticks(rotation=45)
ax = sns.lineplot(x="date_days", y="compound", data=df)
ax.set_title('Sentiment on Danish Christmas Twitter')
ax.set_ylabel('Average Sentiment')
ax.set_xlabel('Date')
plt.savefig('plots/sentiment_compound.png', bbox_inches="tight")

plt.xticks(rotation=45)
ax = sns.lineplot(x="date_days", y="neg", data=df)
ax.set_title('Negative Sentiment on Danish Christmas Twitter')
ax.set_ylabel('Average Negative Sentiment')
ax.set_xlabel('Date')
plt.savefig('plots/sentiment_neg.png', bbox_inches="tight")

plt.xticks(rotation=45)
ax = sns.lineplot(x="date_days", y="pos", data=df)
ax.set_title('Positive Sentiment on Danish Christmas Twitter')
ax.set_ylabel('Average Negative Sentiment')
ax.set_xlabel('Date')
plt.savefig('plots/sentiment_pos.png', bbox_inches="tight")