import re
import json
import numpy as np
import pandas as pd

apos_end_pattern = (
    r"'( cause| d| em| ll| m| n| re| s| til| till| twas| ve) (?!')"
)
apos_start_pattern = r" (d |j |l |ol |y )'"
apos_double_pattern = r" ' n ' "


def load_dict(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def load_index_word_map(
        word2ind_filename='word2ind',
        ind2word_filename='ind2word'
        ):
    word2ind = load_dict(word2ind_filename)
    ind2word = load_dict(ind2word_filename)
    return word2ind, ind2word


def bi_pad_space(mobject):
    return ' ' + mobject.group(0) + ' '


def fix_ing(mobject):
    return mobject.group(0).replace("'", 'g')


def recover_end_apos(mobject):
    return mobject.group(0)[:1] + mobject.group(0)[2:]


def recover_start_apos(mobject):
    return mobject.group(0)[:-2] + mobject.group(0)[-1:]


def recover_double_apos(mobject):
    return mobject.group(0)[:2] + mobject.group(0)[3] + mobject.group(0)[-2:]


def data_curating(dat):
    dat.loc[:, 'cleaned_text'] = dat.text.str.lower()
    dat.loc[:, 'cleaned_text'] = dat.cleaned_text.str.replace(
        "''cause",
        "'cause"
    )
    dat.loc[:, 'cleaned_text'] = dat.cleaned_text.str.replace("n't", " n't")
    dat.loc[:, 'cleaned_text'] = dat.cleaned_text.str.replace("''", '"')
    dat.loc[:, 'cleaned_text'] = dat.cleaned_text.str.replace(
        r'[\n!"\(\),-.0-9:?\[\]]',
        bi_pad_space
    )
    dat.loc[:, 'cleaned_text'] = dat.cleaned_text.str.replace("'", " ' ")
    # in' to ing
    dat.loc[:, 'cleaned_text'] = dat.cleaned_text.str.replace(
        r"\w+in'$|\w+in'\s",
        fix_ing
    )
    # recover n't
    dat.loc[:, 'cleaned_text'] = dat.cleaned_text.str.replace(
        r" n ' t ",
        " n't "
    )
    # recover 'cause, 'd, 'em, 'll, 'm, 'n, 're, 's, 'til, 'till, 'twas
    dat.loc[:, 'cleaned_text'] = dat.cleaned_text.str.replace(
        apos_end_pattern,
        recover_end_apos
    )
    # recover d', j', l', ol', y'
    dat.loc[:, 'cleaned_text'] = dat.cleaned_text.str.replace(
        apos_start_pattern,
        recover_start_apos
    )
    # recover 'n'
    dat.loc[:, 'cleaned_text'] = dat.cleaned_text.str.replace(
        apos_double_pattern,
        recover_double_apos
    )
    dat.loc[:, 'cleaned_text'] = dat.cleaned_text.str.replace(r' {2,}', ' ')
    dat.loc[:, 'cleaned_text'] = dat.cleaned_text.str.strip()
    return dat.loc[:, 'cleaned_text']


def data_cleaning(dat):
    dat.loc[:, 'text_list'] = dat.cleaned_text.str.split(' ')
    return dat.loc[:, 'text_list']


def input_cleaning(sentence):
    sentence = sentence.lower()
    sentence = sentence.replace("''cause", "'cause")
    sentence = sentence.replace("n't", " n't")
    sentence = sentence.replace("''", '"')
    sentence = re.sub(r'[\n!"\(\),-.0-9:?\[\]]', bi_pad_space, sentence)
    sentence = re.sub(r"\w+in'$|\w+in'\s", fix_ing, sentence)
    sentence = sentence.replace("'", " ' ")
    # recover 'cause, 'd, 'em, 'll, 'm, 'n, 're, 's, 'til, 'till, 'twas
    sentence = re.sub(apos_end_pattern, recover_end_apos, sentence)
    # recover d', j', l', ol', y'
    sentence = re.sub(apos_start_pattern, recover_start_apos, sentence)
    # recover 'n'
    sentence = re.sub(apos_double_pattern, recover_double_apos, sentence)
    sentence = sentence.replace(" n ' t ", " n't ")
    sentence = re.sub(r' {2,}', ' ', sentence)
    sentence = sentence.strip()
    sentence = sentence.split(' ')
    return sentence


unapos_end_pattern = r" '(d|em|ll|m|n|re|s|til|till|twas|ve) "
unapos_start_pattern = r" (d|j|l|ol|y)' "
unapos_double_pattern = r" 'n' "


def formatting(sentence):
    sentence = re.sub(r'[\(\[] ', lambda x: x.group(0)[:-1], sentence)
    sentence = re.sub(r' [\)\].!?,:]', lambda x: x.group(0)[1:], sentence)
    sentence = re.sub(
        r'" .+ "',
        lambda x: x.group(0).replace('" ', '"').replace(' "', '"'),
        sentence
    )
    sentence = re.sub(
        r'^[\[\("]?\w|[?.!]"? \w',
        lambda x: x.group(0).upper(),
        sentence
    )
    sentence = re.sub(r' i ', ' I ', sentence)
    sentence = re.sub(
        unapos_start_pattern,
        lambda x: x.group(0)[:-1],
        sentence
    )
    sentence = re.sub(unapos_end_pattern, lambda x: x.group(0)[1:], sentence)
    sentence = re.sub(
        unapos_double_pattern,
        lambda x: x.group(0)[1:-1],
        sentence
    )
    sentence = sentence.replace(" n't ", "n't ")
    # doin ' => doin'
    sentence = re.sub(
        r"(\win) (' )",
        lambda m: m.group(1)+m.group(2),
        sentence
    )
    sentence = re.sub(r' {2,}', ' ', sentence)
    # first letter alignment
    sentence = re.sub(
        r'(\n) (\w)',
        lambda m: m.group(1)+m.group(2).upper(),
        sentence
    )
    sentence = sentence.strip()
    return sentence


def get_unique_tokens(text_sets):
    tokens = set()
    for ind, item in text_sets.iteritems():
        tokens = tokens | item
    return len(tokens), sorted(list(tokens))


def save_unique_tokens(unique_tokens, filename='unique_tokens'):
    # check if all unique
    msg = 'Make sure all tokens unique!'
    assert len(unique_tokens) == len(set(unique_tokens)), msg
    unique_token_series = pd.Series(unique_tokens)
    unique_token_series.to_csv(filename, header=None)


def get_index_word_map(unique_tokens):
    # check if all unique
    msg = 'Make sure all tokens unique!'
    assert len(unique_tokens) == len(set(unique_tokens)), msg
    word2ind = {}
    ind2word = {}
    for ind, word in enumerate(unique_tokens):
        word2ind[word] = ind
        ind2word[ind] = word
    return word2ind, ind2word


def save_dict(dict2save, filename):
    with open(filename, 'w') as f:
        json.dump(dict2save, f)


def save_index_word_map(
        word2ind, ind2word,
        word2ind_filename='word2ind', ind2word_filename='ind2word'):
    save_dict(word2ind, word2ind_filename)
    save_dict(ind2word, ind2word_filename)


def create_emb(ind2word, imported_emb):
    vocab_size = len(ind2word)
    n_fact = imported_emb.shape[1]
    emb = np.zeros((vocab_size, n_fact))
    for i in range(vocab_size):
        word = ind2word[i]
        try:
            emb[i] = imported_emb.loc[word]
        except KeyError:
            emb[i] = np.random.normal(scale=0.6, size=(n_fact,))
    return emb


def save_embedding(embedding, filename='embedding.csv'):
    np.savetxt(filename, embedding, delimiter=',')


def tokenise_cleaned_data(dat, word2ind):
    return dat.apply(lambda words: [word2ind[word] for word in words])


def transform_text(text_list, input_length=10):
    list_length = len(text_list)
    res_list = []
    for i in range(0, list_length-input_length):
        res_list.append(text_list[i:i+input_length+1])
    return res_list


def make_training_samples(pd_series, input_length=10):
    transformed_texts = pd_series.apply(
        transform_text,
        input_length=input_length
    )
    samples = transformed_texts.sum()
    return pd.DataFrame(samples)


def save_training_samples(samples, filename='train.csv'):
    samples.to_csv(filename, index=False, header=None)


def tokenise(word2ind, words):
    return [word2ind.get(word, 0) for word in words]


def detokenise(ind2word, tokens):
    return [ind2word[str(ind)] for ind in tokens]


def load_embedding(filename='embedding.csv'):
    return pd.read_csv(filename, header=None)


def load_training_samples(filename='train.csv'):
    return pd.read_csv(filename, header=None)


def train_valid_split(x, y, test_size=0.3, random_state=43):
    assert len(x) == len(y), 'Feature and label must have same length.'
    np.random.RandomState(seed=random_state)
    length = len(x)
    choices = list(range(length))
    val_choices = np.random.choice(
        choices,
        int(length*test_size),
        replace=False
    ).tolist()
    train_choices = list(set(choices) - set(val_choices))
    return (x[train_choices, :], x[val_choices, :],
            y[train_choices, :], y[val_choices, :])
