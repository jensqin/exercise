import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from nba_torch import NBADataModule, NBAGroupRidge

from utils import root_dir


def train_nba(config, num_epochs=10):
    nba = NBADataModule(
        data_path=os.path.join(root_dir, "data/nba_nw.csv"),
        num_workders=4,
        test_size=0.01,
        # batch_size=config["batch_size"],
    )
    model = NBAGroupRidge(
        lr=config["lr"], weight_decay=list(config["weight_decay"].values())
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


def tune_nba_bohb(num_samples=100, num_epochs=50):
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": {str(x): tune.uniform(0, 0.2) for x in range(6)},
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
        name="tune_nba_asha",
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
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_epochs", type=int, default=30)
    args = parser.parse_args()
    tune_nba_bohb(args.n_samples, args.n_epochs)
