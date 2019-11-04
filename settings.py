import os
from dotenv import find_dotenv, load_dotenv
from data_utilities import db

load_dotenv(find_dotenv(), override=True)

ENGINE_URL = db.load_url_dict(
    prefix=[
        'BASKETBALL_TEST.', 'BASKETBALL_NCAA_DEV.', 
        'FOOTBALL_TEST.', 'FOOTBALL_DEV.'
        ])
