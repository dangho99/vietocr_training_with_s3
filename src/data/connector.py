from src.utils.const import EnvConst

""" MINIO """
from minio import Minio
import numpy as np
import cv2
class MinioConnector:
    client = Minio(endpoint=EnvConst.minio_url,
                    access_key=EnvConst.minio_access_key,
                    secret_key=EnvConst.minio_secret_key,
                    secure=False)
    
    def read_image(self, bucket_name, path_save):
        response = self.client.get_object(bucket_name=bucket_name,object_name=path_save)
        image_data = response.read()
        np_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        return image
 


""" MONGO DB """
from pymongo import MongoClient
class MongoDBManager:
    uri = f"mongodb://{EnvConst.mongodb_username}:{EnvConst.mongodb_password}@{EnvConst.mongodb_host}:{EnvConst.mongodb_port}/"
    client = MongoClient(uri)
    db = client[EnvConst.mongodb_database_name]
    collection = db[EnvConst.mongodb_collection_name]

    def query_all(self):
        return list(self.collection.find())
    
    def query_by_key_value(self, key: str, value):
        query = {key: value}
        return list(self.collection.find(query))

    def insert(self, data):
        if isinstance(data, list):  
            result = self.collection.insert_many(data)
            return result.inserted_ids
        else:  
            result = self.collection.insert_one(data)
            return result.inserted_id

    def delete(self, query: dict):
        result = self.collection.delete_many(query)
        return result.deleted_count

    def close_connection(self):
        self.client.close()


""" MYSQL """
import mysql.connector
import pandas as pd
class MySQLConnector:
    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=EnvConst.mysql_host,
                username=EnvConst.mysql_username,
                password=EnvConst.mysql_password,
                database=EnvConst.mysql_database
            )
            print("Connection to MySQL database successful!")
        except mysql.connector.Error as e:
            print(f"Error connecting to MySQL: {e}")


    def query_as_dataframe(self, query):
        if not self.connection or not self.connection.is_connected():
            raise ConnectionError("Database connection is not established.")
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query)
            rows = cursor.fetchall()

            df = pd.DataFrame(rows)
            cursor.close()
            return df
        except mysql.connector.Error as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame()

    def close(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("MySQL connection closed.")
