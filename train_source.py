import argparse
from non_dp_source_trainer import NonPrivateSourceTrainer
from source_trainer import SourceTrainer
from shot_source_trainer import ShotSourceTrainer
from shot_config import ShotConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="mnist", help="One of the following: office-31, office-home, visda, mnist.")
    parser.add_argument("--source_domain",
                        help="For office-31, one of amazon, dslr or webcam."
                             "For office-home, one of art, clipart, product or real_world."
                             "For visda and mnist not required.")
    parser.add_argument("--backbone", default="alexnet", help="alexnet, resnet, wideresnet")
    parser.add_argument("--batch_size", type=int) # 64 for office-31
    parser.add_argument("--max_physical_batch_size", type=int, help="Physical batch size to use when memory is limited.")
    parser.add_argument("--c", type=float, help="DP clipping norm.")
    parser.add_argument("--e", type=float, default=8, help="DP epsilon.")
    parser.add_argument("--d", type=float, help="DP delta.")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--no_trials", type=int, default=1)

    # trainer = NonPrivateSourceTrainer(dataset=args.data, domain=args.domain, epochs=20, lr=0.01, batch_size=64,
    #                                   test=args.test)

    for i in range(parser.parse_args().no_trials):
        trainer = ShotSourceTrainer(ShotConfig(parser.parse_args(), False), i)

        trainer.train()

    # if args.test:
    #     trainer.test()
