import os
from dotenv import find_dotenv, load_dotenv
import sqlalchemy
import pandas as pd

load_dotenv(find_dotenv(), override=True)

def dbcon(prefix='', drvstr=None, username=None, password=None, host=None, port=None, dbname=None):
    config = {}
    config['drivername'] = os.environ.get(prefix + 'DATABASE_ENGINE') if drvstr is None else drvstr
    config['username'] = os.environ.get(prefix + 'DATABASE_USER') if username is None else username
    config['password'] = os.environ.get(prefix + 'DATABASE_PASSWORD') if password is None else password
    config['host'] = os.environ.get(prefix + 'DATABASE_HOST') if host is None else host
    config['port'] = os.environ.get(prefix + 'DATABASE_PORT') if port is None else port
    config['database'] = os.environ.get(prefix + 'DATABASE_NAME') if dbname is None else dbname
    assert all([x is not None and len(x) > 0 for x in config.values()]), 'Argument Is Missing!'
    return sqlalchemy.create_engine(sqlalchemy.engine.url.URL(**config))

def dbdata(query, con):
    if isinstance(query, str):
        df = pd.read_sql(query, con)
    elif isinstance(query, dict):
        df = {k:pd.read_sql(v, con) for k, v in query.items()}
    else:
        print('Query MUST Be String Or List!')
