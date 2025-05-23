# Database Interaction
from sqlalchemy import Engine, MetaData, select, update, insert
# Update directory to import util
from util.sqlalchemy_tables import DatasetMetadata, DatasetEmbedCols, DatasetTextCols, Users

# Get all of the metadata of a dataset
def get_metadata(engine: Engine, meta: MetaData, table_name: str) -> dict:
    # Make query
    query = (
        select(DatasetMetadata)
            .where(DatasetMetadata.table_name == table_name)
    )
    output = None
    with engine.connect() as conn:
        for row in conn.execute(query):
            output = row
            break
        conn.commit()
        
    if output is None:
        raise Exception("Query failed")    
    
    # Give column names as keys
    record = {}
    for i, col in enumerate(meta.tables[DatasetMetadata.__tablename__].columns.keys()):
        if i < len(output):
            record[col] = output[i]
        
    return record

# Create a new metadata record
def add_metadata(engine: Engine, params: dict) -> None:
    query = (
        insert(DatasetMetadata)
            .values(**params)
    )
    
    with engine.connect() as conn:
        conn.execute(query)

# Get the text columns of a dataset
def get_text_cols(engine: Engine, table_name: str) -> list[str]:
    query = (
        select(DatasetTextCols.col)
            .where(DatasetTextCols.table_name == table_name)
    )
    text_cols = []
    with engine.connect() as conn:
        for row in conn.execute(query):
            text_cols.append(row[0])
        conn.commit()
    if len(text_cols) == 0:
        print("No text columns to process")
        exit(1)
    else:
        return text_cols
    
# Get the embedding columns of a dataset
def get_embed_cols(engine: Engine, meta: MetaData, table_name: str) -> list[str]:
    query = (
        select(DatasetEmbedCols.col)
            .where(DatasetEmbedCols.table_name == table_name)
    )
    embed_cols = []
    with engine.connect() as conn:
        for row in conn.execute(query):
            embed_cols.append(row[0])
        conn.commit()
        
    if len(embed_cols) == 0:
        metadata = get_metadata(engine, meta, table_name)
        if "embed_col" in metadata.keys() and metadata["embed_col"] is not None:
            embed_cols = [metadata["embed_col"]]
    
    return embed_cols
    
# Update metadata that processing is done
def complete_processing(engine: Engine, table_name: str, processing_type: str) -> None:
    query = (
        update(DatasetMetadata)
            .where(DatasetMetadata.table_name == table_name)
            .values({
                f"{ processing_type }_done": True
            })
    )
    
    with engine.connect() as conn:
        conn.execute(query)
        conn.commit()
        
# Set processing to not done while reprocessing
def deactivate_processing(engine: Engine, table_name: str, processing_type: str) -> None:
    query = (
        update(DatasetMetadata)
            .where(DatasetMetadata.table_name == table_name)
            .values({
                f"{ processing_type }_done": False
            })
    )
    
    with engine.connect() as conn:
        conn.execute(query)
        conn.commit()
        
# Update metadata to complete reprocessing
def complete_reprocessing(engine: Engine, table_name: str) -> None:
    query = (
        update(DatasetMetadata)
            .where(DatasetMetadata.table_name == table_name)
            .values({
                "reprocess_start": False,
                "unprocessed_updates": 0
            })
    )
    
    with engine.connect() as conn:
        conn.execute(query)
        conn.commit()
    
# Get a user record by email
def get_user(engine: Engine, meta: MetaData, email: str) -> dict:
    # Make query
    query = (
        select(Users)
            .where(Users.email == email)
    )
    with engine.connect() as conn:
        for row in conn.execute(query):
            output = row
            break
        conn.commit()
        
    # Give column names as keys
    record = {}
    for i, col in enumerate(meta.tables[Users.__tablename__].columns.keys()):
        record[col] = output[i]
        
    return record

# Update the number of batches completed
def complete_batch(engine: Engine, table_name: str, batch_num: int):
    query = (
        update(DatasetMetadata)
            .where(DatasetMetadata.table_name == table_name)
            .values({
                "batches_done": batch_num
            })
    )
    
    with engine.connect() as conn:
        conn.execute(query)
        conn.commit()