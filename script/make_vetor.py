import ast
import pickle

import pandas as pd
from scipy.sparse import lil_matrix
from sklearn.decomposition import NMF, TruncatedSVD


def get_swap_dict(d):
    return {v: k for k, v in d.items()}


def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj, f)


# 使用するユーザと本の抽出
userlist = []
bookdict = {}
with open("../data/booklist_20200726_131226.csv") as f:
    for s_line in f:
        user, userbooklist = s_line.split(":", 1)
        # 読んだ本が10冊以下の人は含めない
        if len(ast.literal_eval(userbooklist.split("\n")[0])) < 10:
            continue
        userlist.append(user)
        for book in ast.literal_eval(userbooklist.split("\n")[0]):
            if book not in bookdict:
                bookdict[book] = 1
            else:
                bookdict[book] += 1

# 本命：登場回数 の辞書を保存
pickle_dump(bookdict, '../data/bookdict_all_count.pickle')

# 10冊以上読まれている本だけを残す
bookdict_valid = {}
num = 0
for key, value in bookdict.items():
    if value >= 10:
        bookdict_valid[key] = num
        num += 1

# 疎行列の作成
bookset = set([book for book in bookdict_valid.keys()])
mat = lil_matrix((len(userlist), len(bookdict_valid)))
usernum = 0
with open("../data/booklist_20200726_131226.csv") as f:
    for s_line in f:
        user, userbooklist = s_line.split(":", 1)
        # 読んだ本が10冊以下の人は含めない
        if len(ast.literal_eval(userbooklist.split("\n")[0])) < 10:
            continue
        for book in ast.literal_eval(userbooklist.split("\n")[0]):
            if book in bookset:
                mat[usernum, bookdict_valid[book]] = 1
        usernum += 1

# 辞書のkey-value反転
bookdict_swap = get_swap_dict(bookdict_valid)

# 書籍と番号の対応表を保存
pickle_dump(bookdict_swap, '../data/bookdict.pickle')

# NMFによる次元削減
nmf = NMF(n_components=100, init='random', random_state=42)
book100f_nmf = nmf.fit_transform(mat.T)
pd.DataFrame(book100f_nmf).to_pickle('../data/book100f.nmf.pkl')

# SVDによる次元削減
svd = TruncatedSVD(n_components=100, n_iter=5, random_state=42)
book100f_svd = svd.fit_transform(mat.T)
pd.DataFrame(book100f_svd).to_pickle('../data/book100f_svd.pkl')
