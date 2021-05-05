"""Global Settings

This module contains global settings and constants for the project.

"""

from dotenv import find_dotenv, load_dotenv
from bla_python_db_utilities import auth


load_dotenv(find_dotenv(), override=True)

ENGINE_CONFIG = auth.load_url_dict(prefix=["DEV_NBA.",])

# SQL_DIR = "nbastats/sql"
# _SQL_PATH = {"play": "playbyplay.sql", "game": "game.sql"}
SQL_PATH = {
    key: "nbastats/sql/" + value
    for key, value in {
        "play": "playbyplay.sql",
        "game": "game.sql",
        "player": "player.sql",
        "team": "team.sql",
    }.items()
}

S3_FOLDER = "s3://bla-basketball-models/processing/"
# HPARAMS_PATH = "nba/hparams.json"
# MODEL_URL = "NBA/zqin-models/models"
# DATALOADER_URL = "NBA/zqin-models/dataloaders"
# LOG_DIR = "nba/.tensorboard_logs"
