# Database Interaction
from sqlalchemy import Engine, MetaData, select, update, insert
# Update directory to import util
from util.sqlalchemy_tables import DatasetMetadata, DatasetTextCols, Users

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
    