import os
import re
import json
import numpy as np
from keras.models import load_model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout, Flatten
from django.shortcuts import render
from django.conf import settings

dp_model_dir = os.path.join(
    'web',
    settings.STATIC_URL[1:],
    'web',
    'dp-model',
)
dp_model_url = os.path.join(dp_model_dir, 'model20180120_7.h5')
dp_model = load_model(dp_model_url)
dp_model.predict(np.random.randint(0, 10, size=(1, 20)))


def load_dict(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def load_index_word_map(word2ind_filename, ind2word_filename):
    word2ind = load_dict(word2ind_filename)
    ind2word = load_dict(ind2word_filename)
    return word2ind, ind2word


word2ind, ind2word = load_index_word_map(
    word2ind_filename=os.path.join(dp_model_dir, 'word2ind'),
    ind2word_filename=os.path.join(dp_model_dir, 'ind2word')
)
apos_end_pattern = r"'( cause| d| em| ll| m| n| re| s| til| till| twas| ve) (?!')"
apos_start_pattern = r" (d |j |l |ol |y )'"
apos_double_pattern = r" ' n ' "


def input_cleaning(sentence):
    sentence = sentence.lower()
    sentence = sentence.replace("''cause", "'cause")
    sentence = sentence.replace("n't", " n't")
    sentence = sentence.replace("''", '"')
    sentence = re.sub(r'[\n!"\(\),-.0-9:?\[\]]', lambda x: ' '+x.group(0)+' ', sentence)
    sentence = re.sub(r"\w+in'$|\w+in'\s", lambda m: m.group(0).replace("'", 'g'), sentence)
    sentence = sentence.replace("'", " ' ")
    # recover 'cause, 'd, 'em, 'll, 'm, 'n, 're, 's, 'til, 'till, 'twas
    sentence = re.sub(apos_end_pattern, lambda m: m.group(0)[:1]+m.group(0)[2:], sentence)
    # recover d', j', l', ol', y'
    sentence = re.sub(apos_start_pattern, lambda m: m.group(0)[:-2]+m.group(0)[-1:], sentence)
    # recover 'n'
    sentence = re.sub(apos_double_pattern, lambda m: m.group(0)[:2]+m.group(0)[3]+m.group(0)[-2:], sentence)
    sentence = sentence.replace(" n ' t ", " n't ")
    sentence = re.sub(r' {2,}', ' ', sentence)
    sentence = sentence.strip()
    sentence = sentence.split(' ')
    return sentence


def tokenise(words):
    return [word2ind.get(word, 0) for word in words]


def detokenise(tokens):
    return [ind2word[str(ind)] for ind in tokens]


unapos_end_pattern = r" '(d|em|ll|m|n|re|s|til|till|twas|ve) "
unapos_start_pattern = r" (d|j|l|ol|y)' "
unapos_double_pattern = r" 'n' "


def formatting(sentence):
    sentence = re.sub(r'[\(\[] ', lambda x: x.group(0)[:-1], sentence)
    sentence = re.sub(r' [\)\].!?,:]', lambda x: x.group(0)[1:], sentence)
    sentence = re.sub(r'" .+ "', lambda x: x.group(0).replace('" ', '"').replace(' "', '"'), sentence)
    sentence = re.sub(r'^[\[\("]?\w|[?.!]"? \w', lambda x: x.group(0).upper(), sentence)
    sentence = re.sub(r' i ', ' I ', sentence)
    sentence = re.sub(unapos_start_pattern, lambda x: x.group(0)[:-1], sentence)
    sentence = re.sub(unapos_end_pattern, lambda x: x.group(0)[1:], sentence)
    sentence = re.sub(unapos_double_pattern, lambda x: x.group(0)[1:-1], sentence)
    sentence = sentence.replace(" n't ", "n't ")
    # doin ' => doin'
    sentence = re.sub(r"(\win) (' )", lambda m: m.group(1)+m.group(2), sentence)
    sentence = re.sub(r' {2,}', ' ', sentence)
    # first letter alignment
    sentence = re.sub(r'(\n) (\w)', lambda m: m.group(1)+m.group(2).upper(), sentence)
    sentence = sentence.strip()
    return sentence


def semantic_not_end(tokens):
    unequal_square_quotes = tokens.count(word2ind['[']) != tokens.count(word2ind[']'])
    unequal_brackets = tokens.count(word2ind['(']) != tokens.count(word2ind[')'])
    odd_double_quotes = tokens.count(word2ind['"']) % 2 != 0
    nonfinished_sentence = ind2word[str(tokens[-1])] not in ['.', '!', '?', '"', '\n']
    not_end = unequal_square_quotes or unequal_brackets or odd_double_quotes or nonfinished_sentence
    return not_end


def implement(seed_text, n_words, maxlen=10, must_stop=200):
    cleaned = input_cleaning(seed_text)
    padded_input_tokens = tokenise(cleaned)
    res_tokens = [token for token in padded_input_tokens]
    while (n_words > 0 or semantic_not_end(padded_input_tokens)) and must_stop > 0:
        padded_input_tokens = pad_sequences([padded_input_tokens], maxlen=maxlen)
        probs = dp_model.predict(padded_input_tokens)[0]
        probs = probs / np.sum(probs)
        predicted = np.random.choice(list(range(len(probs))), p=probs)
        padded_input_tokens = padded_input_tokens[0].tolist()
        padded_input_tokens.append(predicted)
        res_tokens.append(predicted)
        n_words -= 1
        must_stop -= 1
    detokenised = detokenise(res_tokens)
    return formatting(' '.join(detokenised))


# Create your views here.
def index(request):
    lyrics_input = request.GET.get('lyricsInput')
    if lyrics_input:
        res = implement(seed_text=lyrics_input, n_words=500, maxlen=20)
        # res = res.replace('\n', '<br>')
        context = {'output_lyrics': res}
    else:
        context = {'output_lyrics': 'Tell us a bit what\'s in your mind'}
    return render(request, 'web/index.html', context)
