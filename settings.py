import os
from dotenv import find_dotenv, load_dotenv
from sqlalchemy.engine.url import URL


load_dotenv(find_dotenv(), override=True)

def load_url(prefix='', database=None):
    engine = os.environ.get(f'{prefix}DATABASE_ENGINE')
    if engine is None:
        engine = 'mysql+pymysql'
    user = os.environ.get(f'{prefix}DATABASE_USER')
    password = os.environ.get(f'{prefix}DATABASE_PASSWORD')
    host = os.environ.get(f'{prefix}DATABASE_HOST')
    port = os.environ.get(f'{prefix}DATABASE_PORT')
    if database is None:
        dbname = os.environ.get(f'{prefix}DATABASE_NAME')
    else:
        dbname = database
    url = URL(
        engine, username=user, password=password, 
        host=host, port=port, database=dbname
    )
    return url

def load_url_dict(prefix=''):
    if isinstance(prefix, list):
        config = {x + 'url': load_url(x) for x in prefix}
        return config
    elif isinstance(prefix, str):
        return {prefix + 'url': load_url(prefix)}
    else:
        print('prefix must be either list or str!')

ENGINE_URL = load_url_dict(prefix='BASKETBALL_DEV.')
