from argparse import ArgumentParser

import pandas as pd
import pyro
import torch
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide.guides import AutoLowRankMultivariateNormal
from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from sklearn import metrics
from torch.utils.data.dataloader import DataLoader

from nba_pyro import NBABayesEncoder
from nba_torch import NBADataset
from utils import load_nba, pyro_summary, transform_to_tensors


def train_nba(config, num_epochs=10):
    df_train, df_val = load_nba(split_mode="test", test=0.2)
    train = NBADataset(df_train)
    train_loader = DataLoader(train, batch_size=32)
    model = NBABayesEncoder(scale_global=config["scale_global"])
    guide = AutoLowRankMultivariateNormal(model)
    adam = pyro.optim.AdamW({"lr": config["lr"]})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())
    pyro.clear_param_store()

    x_val, y_val = transform_to_tensors(df_val)

    for _ in range(num_epochs):
        for batch in iter(train_loader):
            x_val, y_val = batch
            y_val = torch.flatten(y_val)
            _ = svi.step(x_val, y_val)

        predictive = Predictive(
            model,
            guide=guide,
            num_samples=800,
            return_sites=("fc.weight", "obs", "_RETURN"),
        )
        samples = predictive(x_val)
        pred_summary = pyro_summary(samples)
        mu = pred_summary["_RETURN"]
        # yhat = pred_summary["obs"]
        # predictions = pd.DataFrame(
        #     {
        #         "mu_mean": mu["mean"],
        #         "mu_perc_5": mu["5%"],
        #         "mu_perc_95": mu["95%"],
        #         "y_mean": yhat["mean"],
        #         "y_perc_5": yhat["5%"],
        #         "y_perc_95": yhat["95%"],
        #         "true_y": y_val,
        #     }
        # )
        mse = metrics.mean_squared_error(mu["mean"], y_val)
        tune.report(score=mse)


def tune_nba_pyro(num_samples=100, num_epochs=50):
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "scale_global": tune.choice([1, 2, 3, 4]),
    }
    algo = TuneBOHB(max_concurrent=4)
    scheduler = HyperBandForBOHB(
        max_t=num_epochs, time_attr="training_iteration", reduction_factor=3
    )
    reporter = tune.CLIReporter(
        parameter_columns=["lr"], metric_columns=["score", "training_iteration"],
    )
    analysis = tune.run(
        tune.with_parameters(train_nba, num_epochs=num_epochs),
        config=config,
        metric="score",
        mode="min",
        num_samples=num_samples,
        search_alg=algo,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_nba_pyro",
        # stop={"training_iteration": num_epochs},
    )
    print("Best hyperparameters are: ", analysis.best_config)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_sample", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=5)
    args = parser.parse_args()
    tune_nba_pyro(args.n_sample, args.max_epochs)
