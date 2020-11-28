from ray import tune
from ray.tune.suggest import HyperOptSearch
from ray.tune.schedulers import PopulationBasedTraining

class RegTrainable(tune.Trainable):
    def setup(self, config):
        # config (dict): A dict of hyperparameters
        self.x = 0
        self.a = config["a"]
        self.b = config["b"]

    def step(self):  # This is called iteratively.
        score = max(self.x, self.a, self.b)
        self.x += 1
        return {"score": score}

