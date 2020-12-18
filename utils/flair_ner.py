"""
"""

from typing import List

from flair.data import Sentence, Token
from flair.models import SequenceTagger

from utils.utils import add_def_args


def custom_token_flair_sent(tokenlist):

    def custom_flair_tokens(text: str, tokenlist) -> List[Token]:
        """
        Does not tokenize is simply a function for giving a predefined
        tokenlist to flair
        """
        # convert each token to a flair token
        tokens: List[Token] = [Token(token) for token in tokenlist]
        return tokens
    custom_tokens = add_def_args(custom_flair_tokens, {'tokenlist': tokenlist})
    return Sentence(" ", use_tokenizer=custom_tokens)


def flair_tagger(sentence, lang: str, tagging='ner', tagger=None):
    """
    sentence (str|list): string either as a list of tokens or as a text
    lang (str): a two letter language code
    tagging ('ner'|'pos'): whether is should do POS-tagging or NER tagging
    tagger (None|str): if tagger is given lang and tagging is ignored and the
    specified tagger is used.
    """
    if tagger is None:
        tagger = lang + '-' + tagging

    try:
        tagger = SequenceTagger.load(tagger)
    except FileNotFoundError:
        raise ValueError(f'The tagger {tagger} was not found. it is likely that \
                           make sure that the tagger is valid.')

    if isinstance(sentence, list):
        sentence = custom_token_flair_sent(sentence)
    elif isinstance(sentence, str):
        sentence = Sentence(sentence)
    elif not isinstance(sentence, Sentence):
        raise ValueError(f'sentence should a string or a list \
                           not a type {type(sentence)}')
    return [(tok.text, tok.get_tag(tagging).values, tok.get_tag(tagging).score)
            for tok in sentence]


def flair_tagger_ttt(dfs: list, langs, tagging: str = "ner", **kwargs) -> list:
    """
    dfs (list): list of dataframes
    tagging ('pos', 'ner')

    Examples:
    """

    def __flair_add_tag(df):
        sentence = custom_token_flair_sent(df['token'])
        tagger.predict(sentence)
        if len(sentence) < len(df['token']):  # very rare
            tagged = [(tok.text, tok.get_tag(tagging).value) for tok in
                      sentence]
            # add missing tokens (typically special character or similar)
            while len(tagged) < len(df["token"]):
                for i, t in enumerate(df["token"]):
                    if i == len(tagged):
                        tagged = tagged[:i] + [(t, "O")]
                    if t != tagged[i][0]:
                        tagged = tagged[:i] + \
                            [(t, "O")] + tagged[i:]
                        break
            ner = [t[1] for t in tagged]
        else:
            ner = [tok.get_tag(tagging).value for tok in sentence]
        df['ner'] = ner
        return df

    if isinstance(langs, str):
        tagger = langs + '-' + tagging
        tagger = SequenceTagger.load(tagger)
        return [__flair_add_tag(df) for df in dfs]
    else:
        res = []
        for i, df in enumerate(dfs):
            if i == 0:
                lang = langs[0]
                tagger = SequenceTagger.load(lang + '-' + tagging)
            elif langs[i] != lang:
                lang = langs[i]
                tagger = SequenceTagger.load(lang + '-' + tagging)
            res.append(__flair_add_tag(df))
        return res


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
