from . import utils
from argparse import Namespace
from datetime import datetime


def run(args: Namespace) -> None:

    # load data
    samples, y = utils.DATASETS[args.dataset_name].get_data()

    # split data
    samples_train, _, y_train, _ = utils.train_test_split(samples, y)

    # train model
    start_time = datetime.now()
    model = utils.MODELS[args.model_name]
    model.fit(samples_train, y_train)
    elapsed_time = datetime.now() - start_time
    if not args.quiet:
        print(f"\n## Training time: {elapsed_time}")

    # evaluate
    y_pred_train = model.predict(samples_train)
    df_scores = utils.evaluate(y_train, y_pred_train)
    if not args.quiet:
        print("\n## Training scores:")
        print(df_scores)

    # save results
    utils.save_model(args.dataset_name, args.model_name)
    utils.save_scores(df_scores, args.dataset_name, args.model_name, "train")


if __name__ == "__main__":
    args = utils.get_args()
    run(args)
