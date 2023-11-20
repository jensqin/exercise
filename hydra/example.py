import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None)
def my_app(cfg: DictConfig) -> None:
    print(cfg)
    print(cfg.db.driver)

if __name__ == "__main__":
    my_app()
