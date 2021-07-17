from . import utils
from argparse import Namespace


def run(args: Namespace) -> None:

    # load data
    samples, y = utils.DATASETS[args.dataset_name].get_data()

    # split data
    _, samples_test, _, y_test = utils.train_test_split(samples, y)

    # load model
    dir = f"{utils.RESULTS_DIR}/{args.dataset_name}/{args.model_name}"
    model = utils.MODELS[args.model_name]()
    model.load(dir)

    # evaluate
    y_pred_test = model.predict(samples_test)
    df_scores = utils.evaluate(y_test, y_pred_test)
    if not args.quiet:
        print("\n## Testing scores:")
        print(df_scores)

    # save results
    utils.save_scores(df_scores, dir, "test")


if __name__ == "__main__":
    args = utils.get_args()
    run(args)
