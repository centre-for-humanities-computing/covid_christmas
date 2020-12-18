"""
A selection of function for dealing with twitter data.
"""

import re
import numpy as np


def twitter_format(dfs):
    """
    call this function to deal with emoticons, hashtags (#) and at (@)
    it will add to 'ner' TWITTER_USER and HASHTAG
    and will change POS tag for emojis to SYM_EMOJI
    """

    # regex replies (@something)
    # twitter username can be letters or numbers
    at_pattern = r'\B@\w*[a-zA-Z0-9]+\w*'

    # regex hashtags (#something)
    # twitter hashtag can't start with a number
    hash_pattern = r'\B#\w*[a-zA-Z]+\w*'

    # regex emojis
    # unicode characters in certain range
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)

    # regex url
    # this is a monster, but works wonders
    url_pattern = re.compile(
        r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')

    for i, df in enumerate(dfs):
        # ID of rows containing nothing but #
        hashtag_idx = np.array(df['token'] == "#")
        # ID of rows after a tag
        idx_offset = np.insert(hashtag_idx, 0, False)[:-1]
        # add hashtag to token
        df.loc[idx_offset, ['token']] = '#' + df.loc[idx_offset, ['token']]
        df.loc[idx_offset, ['lemma']] = '#' + df.loc[idx_offset, ['lemma']]
        # remove hashtag columns
        df = df.loc[np.invert(hashtag_idx)]
        # tag remaining hashtags with the re pattern
        hashtag_idx = np.array(df['token'].str.match(hash_pattern))
        # save ner tag
        df.loc[hashtag_idx, ['ner']] = "HASHTAG"

        # same procedure as above, only with @
        at_idx = np.array(df['token'] == "@")
        idx_offset = np.insert(at_idx, 0, False)[:-1]
        df.loc[idx_offset, ['token']] = '@' + df.loc[idx_offset, ['token']]
        df.loc[idx_offset, ['lemma']] = '@' + df.loc[idx_offset, ['lemma']]
        df = df.loc[np.invert(at_idx)]
        at_idx = np.array(df['token'].str.match(at_pattern))
        df['ner'][at_idx] = "TWITTER_USER"

        # match emojis
        emoji_idx = np.array(df['token'].str.match(emoji_pattern))
        df.loc[emoji_idx, ['upos']] = "SYM_EMOJI"

        # match urls
        url_idx = np.array(df['token'].str.match(url_pattern))
        df.loc[url_idx, ['ner']] = "URL"
        yield(df)
