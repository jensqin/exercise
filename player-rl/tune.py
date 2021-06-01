import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from modules.core import NBADataModule
from models.ridge import NBARidge
from models.mlr import NBAMixedLogit, NBARidgeMLR, NBAMLRShootEmb
from models.dcn import NBADCN
from models.former import NBATransformer, NBAShootTF
from utils import root_dir


def train_nba(config, num_epochs=10):
    nba = NBADataModule(
        data_path=os.path.join(root_dir, "data/nba_nw.csv"),
        num_workders=4,
        batch_size=32,
        test_size=0.01,
        betloss=True
        # batch_size=config["batch_size"],
    )
    model = NBARidgeMLR(
        lr=config["lr"],
        weight_decay=list(config["weight_decay"].values()),
        betloss=True,
    )
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCallback({"loss": "val_loss"}, on="validation_end"),
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=5,
                verbose=False,
                mode="min",
            ),
        ],
    )
    trainer.fit(model, datamodule=nba)


def tune_nba_torch(num_samples=100, num_epochs=50):
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": {str(x): tune.uniform(0, 0.2) for x in range(6)},
        # "weight_decay": {str(x): 0.05 for x in range(6)},
    }
    algo = TuneBOHB(max_concurrent=4)
    scheduler = HyperBandForBOHB(
        max_t=num_epochs, time_attr="training_iteration", reduction_factor=3
    )
    reporter = tune.CLIReporter(
        parameter_columns=["lr"], metric_columns=["loss", "training_iteration"],
    )
    analysis = tune.run(
        tune.with_parameters(train_nba, num_epochs=num_epochs),
        config=config,
        metric="loss",
        mode="min",
        num_samples=num_samples,
        search_alg=algo,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_nba_torch",
        # stop={"training_iteration": num_epochs},
    )
    print("Best hyperparameters are: ", analysis.best_config)


if __name__ == "__main__":
    # # Execute 10 trials using HyperOpt and stop after 10 iterations
    # analysis = tune.run(
    #     train_nba,
    #     config=config,
    #     search_alg=hyperopt,
    #     scheduler=asha,
    #     num_samples=10,
    #     stop={"training_iteration": 10},
    #     verbose=2,
    # )
    # Best hyperparameters loss=1.3636797666549683 are:
    # {
    #     "lr": 0.057632504767510764,
    #     "weight_decay": {
    #         "0": 0.06393286379310652,
    #         "1": 0.017593797533765866,
    #         "2": 0.17836322927053908,
    #         "3": 0.024618031166007873,
    #         "4": 0.16948488333322295,
    #         "5": 0.088538960032512,
    #     },
    # }
    parser = ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--n_epochs", type=int, default=15)
    args = parser.parse_args()
    tune_nba_torch(args.n_samples, args.n_epochs)
    # 2018 mse: 1.36313
    # Best hyperparameters are:
    # {
    #     "lr": 0.0006394159690362793,
    #     "weight_decay": {
    #         "0": 0.08390024504310523,
    #         "1": 0.1870588988483033,
    #         "2": 0.19103583835202897,
    #         "3": 0.023985152534154463,
    #         "4": 0.1971911986048563,
    #         "5": 0.021063980012231264,
    #     },
    # }
    # SGD loss: 1.35446
    # {
    #     "lr": 0.0020747915677990433,
    #     "weight_decay": {
    # "0": 0.02090030251079458,
    # "1": 0.08499998457075844,
    # "2": 0.12153418314918588,
    # "3": 0.1272122955065476,
    # "4": 0.03716296686196379,
    # "5": 0.11646519017723754,
    #     },
    # }
