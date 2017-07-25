from datetime import datetime

import sentiment
from sklearn.model_selection import ParameterGrid


def grid_search():
    param_grid = {}
    param_grid["embeddings"] = [("data/pol/orth", "w2v_allwiki_nkjp300_300"),
                                ("data/pol/lemma", "w2v_allwiki_nkjp300_300"),
                                ("data/pol/fasttext", "wiki.pl")]
    param_grid["optim"] = ["adam", "adagrad"]
    param_grid["wd"] = [1e-5, 1e-4, 1e-3]
    param_grid['reweight'] = [True, False]
    grid = ParameterGrid(param_grid)

    filename = "results_{date:%Y%m%d_%H%M}.csv".format(date=datetime.now())
    for params in grid:
        print(params)
        with open(filename, "a") as myfile:
            myfile.write(str(params) + ",")
        sentiment.main(params)

if __name__ == "__main__":
    grid_search()
