from . import utils
from argparse import Namespace
from datetime import datetime


def run(args: Namespace) -> None:

    model_names = list(utils.MODELS.keys())
    if args.model_name != "all":
        model_names = [args.model_name]

    for i, model_name in enumerate(model_names):
        args.model_name = model_name

        print("\n", "-" * 80)
        print(f"## Iteration {i}")
        print(f"dataset_name: {args.dataset_name}")
        print(f"model_name: {args.model_name}")

        _train(args)
        _predict(args)


def _train(args: Namespace) -> None:

    # load data
    samples, y = utils.DATASETS[args.dataset_name].get_data()

    # split data
    samples_train, _, y_train, _ = utils.train_test_split(samples, y)

    # train model
    if not args.quiet:
        print("\n## Training...")
    start_time = datetime.now()
    model = utils.MODELS[args.model_name]()
    model.fit(samples_train, y_train)
    elapsed_time = datetime.now() - start_time
    if not args.quiet:
        print(f"elapsed time: {elapsed_time}")

    # evaluate
    if not args.quiet:
        print("\n## Evaluating...")
    y_pred_train = model.predict(samples_train)
    scores = utils.evaluate(y_train, y_pred_train)
    if not args.quiet:
        print(f"training scores: {scores}")

    # save results
    dir = f"{utils.RESULTS_DIR}/{args.dataset_name}/{args.model_name}"
    model.save(dir)
    utils.save_scores(scores, dir, "train")


def _predict(args: Namespace) -> None:

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
    scores = utils.evaluate(y_test, y_pred_test)
    if not args.quiet:
        print(f"testing scores: {scores}")

    # save results
    utils.save_scores(scores, dir, "test")


if __name__ == "__main__":
    args = utils.get_args()
    run(args)
