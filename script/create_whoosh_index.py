import os.path
import pickle

from sudachipy import dictionary, tokenizer
from whoosh.analysis import StandardAnalyzer
from whoosh.fields import ID, KEYWORD, NUMERIC, STORED, TEXT, Schema
from whoosh.index import create_in


def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data


# 書籍と登場回数の辞書を読み込み
bookdict_all = pickle_load('../data/bookdict_all_count.pickle')

# 登場回数でソート
bookdict_all_sort = sorted(
    bookdict_all.items(), key=lambda x: x[1], reverse=True)

# 10回以上に限定
bookdict_all_sort_upper10 = []
for i in bookdict_all_sort:
    if i[1] >= 10:
        bookdict_all_sort_upper10.append(i)

remove_words = set({"(", ")", "（", "）", "[", "]",
                    "「", "」", "+", "-", "*", "$",
                    "'", '"', "、", ".", "”", "’",
                    ":", ";", "_", "/", "?", "!",
                    "。", ",", "=", "＝", " ", '『', '』'})

# sudachiの設定
tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C

# １文字を許可するためcontentのstoplistを無効化
schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True, analyzer=StandardAnalyzer(stoplist=None)),
                count=NUMERIC(stored=True, sortable=True))
if not os.path.exists("../data/heroku_index"):
    os.mkdir("../data/heroku_index")
ix = create_in("../data/heroku_index", schema)

# index作成
writer = ix.writer()
for num in range(len(bookdict_all_sort_upper10)):
    titlewords = set([m.surface() for m in tokenizer_obj.tokenize(
        bookdict_all_sort_upper10[num][0], mode)])
    titlewords = titlewords.union(set([m.normalized_form(
    ) for m in tokenizer_obj.tokenize(bookdict_all_sort_upper10[num][0], mode)]))
    titlewords = titlewords.union(set([m.dictionary_form(
    ) for m in tokenizer_obj.tokenize(bookdict_all_sort_upper10[num][0], mode)]))
    writer.add_document(title=bookdict_all_sort_upper10[num][0], content=" ".join(list(titlewords - remove_words)),
                        count=bookdict_all_sort_upper10[num][1])

writer.commit()
