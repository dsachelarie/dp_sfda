import argparse
import wandb
import numpy as np
import sys
from shot_config import ShotConfig
from shot_target_trainer import ShotTargetTrainer


if __name__ == '__main__':
     parser = argparse.ArgumentParser()

     parser.add_argument("--dataset", default="visda",
                         help="One of the following: office-31, office-home, visda.")
     parser.add_argument("--source_domain",
                         help="For office-31, one of amazon, dslr or webcam."
                              "For office-home, one of art, clipart, product or real_world."
                              "For visda not required.")
     parser.add_argument("--target_domain",
                         help="For office-31, one of amazon, dslr or webcam."
                              "For office-home, one of art, clipart, product or real_world."
                              "For visda not required.")
     parser.add_argument("--backbone", default="alexnet", help="alexnet, resnet, wideresnet")
     parser.add_argument("--batch_size", type=int) # 16 for generator, 64 for da
     parser.add_argument("--source_e", default=8, help="A positive number or -1 for no privacy")
     parser.add_argument("--max_physical_batch_size", type=int,
                         help="Physical batch size to use when memory is limited.")
     parser.add_argument("--c", type=float, help="DP clipping norm.")
     parser.add_argument("--e", type=float, default=8, help="DP epsilon of target, a positive number or -1 for no privacy.")
     parser.add_argument("--d", type=float, help="DP delta.")
     parser.add_argument("--lr", type=float)
     parser.add_argument("--epochs", type=int)
     parser.add_argument("--no_trials", type=int, default=1)
     parser.add_argument("--use_few_shot", action="store_true",)
     # parser.add_argument("--da_lr", type=float, default=0.01)
     # parser.add_argument("--da_epochs", type=int, default=20000)
     # parser.add_argument("--test", action="store_true", help="Test mode, uses 20% of the data as test set.")

     # trainer = TargetTrainer(train_data, dataset=args.data, domain=args.domain,
     #                         batch_size=args.batch_size,
     #                         max_physical_batch_size=min(args.batch_size, args.physical_batch_size),
     #                         c=args.c, epsilon=args.e, lr=args.lr, epochs=args.epochs,
     #                         delta=args.d)

     # world_size = int(os.getenv('WORLD_SIZE'))
     # rank = int(os.getenv('RANK'))
     # local_rank = int(os.getenv('LOCAL_RANK'))

     # torch.distributed.init_process_group(
     #     backend='nccl',
     #     world_size=world_size,
     #     rank=rank,
     #     timeout=datetime.timedelta(
     #         minutes=60
     #     ),  # Transformations can be slow, increase timeout
     # )
     accuracies = []

     for i in range(parser.parse_args().no_trials):
          trainer = ShotTargetTrainer(ShotConfig(parser.parse_args(), True, trial=i))

          # accuracies.append(trainer.test(msg="source-only"))
          trainer.test(msg="source-only")
          # trainer.test()

          trainer.train()

          accuracies.append(trainer.test(msg="test"))

          wandb.finish()

     print(accuracies, file=sys.stderr)
     print(np.mean(accuracies), file=sys.stderr)

     # trainer.train_synthetic_samples_generator()
     # trainer.test_synthetic_samples_generator()
     # trainer.train_feature_extractor()
     # trainer.test_feature_extractor()

     # if args.test:
     #     trainer.final_test()

     # torch.distributed.destroy_process_group()
