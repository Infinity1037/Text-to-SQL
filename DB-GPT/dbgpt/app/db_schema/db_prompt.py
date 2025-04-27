import mysql.connector
import dotenv
import os

dotenv.load_dotenv()


def connect_to_db():
    return mysql.connector.connect(
        host=os.getenv("LOCAL_DB_HOST"),
        user=os.getenv("LOCAL_DB_USER"),
        password=os.getenv("LOCAL_DB_PASSWORD"),
        database="dbgpt"
    )


def get_all_tables():
    with connect_to_db() as db:
        with db.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            return [table[0] for table in tables]


def get_all_columns(table_name):
    with connect_to_db() as db:
        with db.cursor() as cursor:
            cursor.execute(f"SHOW COLUMNS FROM {table_name}")
            columns = cursor.fetchall()
            return [column[0] for column in columns]










