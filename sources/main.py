from . import predict, train, utils
from argparse import Namespace


def run(args: Namespace) -> None:
    train.run(args)
    predict.run(args)


if __name__ == "__main__":
    args = utils.get_args()
    run(args)
