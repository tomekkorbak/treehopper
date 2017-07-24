import sentiment
from sklearn.model_selection import ParameterGrid

def grid_search():
    param_grid = {}
    param_grid["embeddings"] = [("data/pol/orth","w2v_allwiki_nkjp300_300"),
                                ("data/pol/lemma","w2v_allwiki_nkjp300_300"),
                                ("data/pol/fasttext","wiki.pl")]
    param_grid["optim"] = ["adam","adagrad"]
    param_grid["wd"] = [1e-5,1e-4,1e-3]
    grid = ParameterGrid(param_grid)

    for params in grid:
        print(params)
        with open("results.csv", "a") as myfile:
            myfile.write(str(params) + ",")
        sentiment.main(params)

if __name__ == "__main__":
    grid_search()