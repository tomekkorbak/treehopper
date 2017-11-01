import argparse

from treehopper.src.config import parse_args
from treehopper.src.evaluate.sentiment import main
from treehopper.src.predict.predict_results import predict, save_submission

if __name__ == '__main__':
    args = parse_args()
    model_file_name = 'models/sample_model/model_0.pth'
    if args.train:
        _,_,model_file_name = main()
        print("Best model saved in file {}".format(model_file_name))
    if args.predict:
        predictions = predict(model_file_name)
        save_submission(predictions, args.predict)