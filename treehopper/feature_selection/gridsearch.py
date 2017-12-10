from datetime import datetime

from sklearn.model_selection import ParameterGrid

import train


def grid_search():
    param_grid = {}
    param_grid["embeddings"] = [
        ("data/pol/orth", "w2v_allwiki_nkjp300_300"),
        ("data/pol/lemma", "w2v_allwiki_nkjp300_300"),
        ("resources/pol/fasttext", "wiki.pl")
    ]
    param_grid["optim"] = ["adam", "adagrad"]
    param_grid['reweight'] = [True, False]
    grid = ParameterGrid(param_grid)

    filename = "results/{date:%Y%m%d_%H%M}_results.csv".format(date=datetime.now())
    print('Starting a grid search through {n} parameter combinations'.format(
        n=len(grid)))
    for params in grid:
        print(params)
        with open(filename, "a") as results_file:
            results_file.write(str(params) + ", ")
            max_dev_epoch, max_dev, _ = train.main(params)
            results_file.write('Epoch {epoch}, accuracy {acc:.4f}\n'.format(
                epoch=max_dev_epoch,
                acc=max_dev
            ))

if __name__ == "__main__":
    grid_search()
