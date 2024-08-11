import os
from sqlalchemy import Engine, MetaData, create_engine

def sql_connect() -> tuple[Engine, MetaData]:
    # Connect to default database if no distributed connection
    # Load os.environment variables
    host = os.environ.get("HOST")
    database = os.environ.get("DATABASE")
    port = os.environ.get("PORT")
    username = os.environ.get("DATABASE_USERNAME")
    password = os.environ.get("PASSWORD")

    # Connect to database
    conn_str = "mysql+pymysql://{}:{}@{}:{}/{}".format(
        username, password, host, port, database
    )
            
    engine = create_engine(conn_str)
    meta = MetaData()
    meta.reflect(engine)
    
    return engine, meta