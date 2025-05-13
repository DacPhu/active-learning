import os
import warnings

import pytorch_lightning as pl
import setproctitle

from core.configs import cfg
from core.train_learners import Test
from core.utils.misc import mkdir, parse_args

warnings.filterwarnings('ignore')


def main():
    args = parse_args()

    output_dir = os.path.join(cfg.SAVE_DIR, 'test')
    if output_dir:
        mkdir(output_dir)

    setproctitle.setproctitle(f'{args.proctitle}')

    pl.seed_everything(cfg.SEED, workers=True)

    learner = Test(cfg)

    # init trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=cfg.SOLVER.GPUS,
        precision=16,
        detect_anomaly=True)

    # start training
    trainer.test(learner)


if __name__ == '__main__':
    main()