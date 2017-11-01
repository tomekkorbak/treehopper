from datetime import datetime

from sklearn.model_selection import ParameterGrid

from treehopper.src.evaluate import sentiment


def grid_search():
    param_grid = {}
    param_grid["embeddings"] = [
        # ("data/pol/orth", "w2v_allwiki_nkjp300_300"),
        # ("data/pol/lemma", "w2v_allwiki_nkjp300_300"),
        ("data/pol/fasttext", "wiki.aa")
    ]
    # param_grid["optim"] = ["adam", "adagrad"]
    param_grid["wd"] = [0]
    # param_grid['reweight'] = [True, False]
    param_grid['mem_dim'] = [300, 400]
    param_grid['recurrent_dropout_h'] = [0.001]
    param_grid['recurrent_dropout_c'] = [0.1]
    param_grid['zoneout_choose_child'] = [True]
    param_grid['common_mask'] = [True]
    param_grid['folds'] = [5]
    param_grid['emblr'] = [0.2]
    # param_grid['name'] = ['{date:%Y%m%d_%H%M}'.format(date=datetime.now())]
    grid = ParameterGrid(param_grid)

    filename = "results/{date:%Y%m%d_%H%M}_results.csv".format(date=datetime.now())
    print('Starting a grid search through {n} parameter combinations'.format(
        n=len(grid)))
    for params in grid:
        print(params)
        with open(filename, "a") as results_file:
            results_file.write(str(params) + ", ")
            max_dev_epoch, max_dev = sentiment.main(params)
            results_file.write('Epoch {epoch}, accuracy {acc:.4f}\n'.format(
                epoch=max_dev_epoch,
                acc=max_dev
            ))

if __name__ == "__main__":
    grid_search()
