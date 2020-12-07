import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from nba_torch import NBADataModule, NBAGroupRidge


def train_nba(config, num_epochs=10):
    nba = NBADataModule(batch_size=config["batch_size"])
    model = NBAGroupRidge(
        lr=config["lr"], weight_decay=list(config["weight_decay"].values())
    )
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[TuneReportCallback({"loss": "val_loss"}, on="validation_end")],
    )
    trainer.fit(model, datamodule=nba)


def tune_nba_bohb(num_samples=100, num_epochs=30):
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": {str(x): tune.uniform(0, 0.2) for x in range(6)},
    }
    algo = TuneBOHB(max_concurrent=4)
    scheduler = HyperBandForBOHB(
        max_t=num_epochs, time_attr="training_iteration", reduction_factor=4
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
    )
    print("Best hyperparameters are: ", analysis.best_config)


if __name__ == "__main__":
    # config = {f"weight_decay_{x}": tune.loguniform(1e-2, 0.5) for x in range(6)}
    # config.update({"lr": tune.loguniform(1e-3, 0.1)})
    # hyperopt = OptunaSearch(metric="score", mode="min")
    # asha = ASHAScheduler(metric="score", mode="min")
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
    tune_nba_bohb()
