import sqlite3
from matplotlib.table import table
import pandas as pd
import os

import openmeteo_requests
import requests_cache
from retry_requests import retry

'''
meteo_db.py, par Nicolas Vigneault janvier 2026
Ce script fournit une classe MeteoDB pour gérer le stockage et la récupération des données météorologiques
à l'aide de SQLite et de l'API Open-Meteo.
Il inclut des fonctionnalités pour créer des tables dynamiques basées sur la latitude et la longitude,
insérer et mettre à jour des enregistrements horaires, et récupérer les données manquantes depuis l'API Open-Meteo.

Voir l'exemple d'utilisation à la fin du fichier pour un exemple concret de son utilisation.
'''

class MeteoDB:
    """A class to manage meteorological data storage and retrieval using SQLite and Open-Meteo API."""
    def __init__(self, db_path: str ="./Data/meteo_dbs.db"):
        self.db_path: str = db_path

        # SQLite database setup
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self.db_connection: sqlite3.Connection = sqlite3.connect(self.db_path)
        self.db_connection.row_factory = sqlite3.Row

        # Open-Meteo client setup
        cache = requests_cache.CachedSession('.meteo_cache', expire_after=-1)
        retry_session = retry(cache, retries=5, backoff_factor=0.2)
        self.openmeteo_client = openmeteo_requests.Client(session=retry_session) # type: ignore

    # ---------------- Table naming ----------------
    @staticmethod
    def table_name(latitude: float, longitude: float) -> str:
        """
        Generate a table name based on latitude and longitude.

        Args:
            latitude (float): Latitude of the location.
            longitude (float): Longitude of the location.

        Returns:
            str: A string representing the table name. (e.g., "meteo_46_077700_m71_885300")
        """
        lat_str = f"{latitude:.6f}".replace('.', '_').replace('-', 'm')
        lon_str = f"{longitude:.6f}".replace('.', '_').replace('-', 'm')
        return f"meteo_{lat_str}_{lon_str}"
    
    # ---------------- Ensure table exists ----------------
    def ensure_table(self, latitude : float, longitude: float, variable_names = ["temperature_2m"]) -> str:
        """
        Ensure the SQLite table for the given latitude and longitude exists.
        
        Args:
            latitude (float): Latitude of the location.
            longitude (float): Longitude of the location.
            variable_names (list): List of variable names to ensure as columns in the table.
        Returns:
            str: The name of the table.
        """
        table = self.table_name(latitude, longitude)

        # Create table with timestamp only if missing
        self.db_connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                timestamp TEXT PRIMARY KEY
            )
        """)

        # Add missing columns
        existing_cols = {
            row[1] for row in self.db_connection.execute(
                f"PRAGMA table_info({table})"
            ).fetchall()
        }

        for var in variable_names:
            if var not in existing_cols:
                self.db_connection.execute(f"ALTER TABLE {table} ADD COLUMN {var} REAL")

        self.db_connection.commit()
        return table

    # ---------------- Insert JSON BLOB ----------------
    def insert_hour(self, latitude : float, longitude: float, timestamp : pd.Timestamp, data_dict):
        """
        Insert or update an hourly data record in the SQLite database.
        Args:
            latitude (float): Latitude of the location.
            longitude (float): Longitude of the location.
            timestamp (pd.Timestamp): Timestamp of the data record.
            data_dict (dict): Dictionary containing the weather data to store.
        """
        table = self.ensure_table(latitude, longitude, variable_names=list(data_dict.columns.drop("timestamp")))

        cols = ["timestamp"] + list(data_dict.keys())
        vals = [timestamp] + list(data_dict.values())

        placeholders = ",".join(["?"] * len(vals))
        columns_sql = ",".join(cols)

        self.db_connection.execute(
            f"INSERT OR REPLACE INTO {table} ({columns_sql}) VALUES ({placeholders})",
            vals
        )
        self.db_connection.commit()

    # ---------------- Upsert and Preserve existing data ----------------
    def bulk_upsert(self, latitude : float, longitude: float, data_frame: pd.DataFrame):
        """
        Bulk upsert data from a DataFrame into the SQLite database, preserving existing data.
        
        Args:
            latitude (float): Latitude of the location.
            longitude (float): Longitude of the location.
            data_frame (pd.DataFrame): DataFrame containing the weather data to upsert.
        """
        table = self.ensure_table(latitude, longitude, variable_names=list(data_frame.columns))
        # data_frame = data_frame.copy()
        # data_frame["timestamp"] = pd.to_datetime(data_frame["timestamp"]).tz_convert("UTC")

        # ---------------- Prepare columns for UPSERT ----------------
        update_columns = [col for col in data_frame.columns if col != "timestamp"]

        # Only include DataFrame columns in UPSERT
        col_list = ", ".join(["timestamp"] + update_columns)
        placeholders = ", ".join(["?"] * (1 + len(update_columns)))
        update_set = ", ".join([f"{col} = excluded.{col}" for col in update_columns])

        # Prepare records for executemany
        records = []
        for _, row in data_frame.iterrows():
            record = [str(row["timestamp"])] + [row[col] if pd.notna(row[col]) else None for col in update_columns]
            records.append(tuple(record))

        # ---------------- Step 3: UPSERT ----------------
        sql = f"""
            INSERT INTO {table} ({col_list})
            VALUES ({placeholders})
            ON CONFLICT(timestamp) DO UPDATE SET
                {update_set};
        """
        self.db_connection.executemany(sql, records)
        self.db_connection.commit()
        

    # ---------------- Fetch missing hours from Open-Meteo ----------------
    def fetch_openmeteo(self, lat, lon, start : pd.Timestamp, end : pd.Timestamp, variables = ["temperature_2m"]):
        startDate = (pd.to_datetime(start)-pd.Timedelta(hours=1)).strftime("%Y-%m-%d")
        endDate = pd.to_datetime(end).strftime("%Y-%m-%d")

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": startDate,
            "end_date": endDate,
            "hourly": ",".join(variables),
            "timezone": "auto",
        }

        responses = self.openmeteo_client.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
       
        timestamps = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds = hourly.Interval()),
            inclusive="left"
        )
        
        # Build variable arrays based on requested variables
        variable_arrays = {}
        for i, var_name in enumerate(variables):
            variable_arrays[var_name] = hourly.Variables(i).ValuesAsNumpy()

        # Build DataFrame
        df = pd.DataFrame({"timestamp": timestamps, **variable_arrays})

        # Bulk upsert into DB
        self.bulk_upsert(lat, lon, df)
        
        # # Insert rows
        # for i, ts in enumerate(timestamps):
        #     row = {var: float(variable_arrays[var][i]) for var in variables}
        #     self.insert_hour(lat, lon, str(ts), row)

    # ---------------- Load hourly data with auto-fill ----------------
    def load_hourly(self, latitude : float, longitude : float, start :pd.Timestamp, end :pd.Timestamp, variable_names) -> pd.DataFrame:
        # Ensure table and all columns exist
        table = self.ensure_table(latitude, longitude, variable_names)

        rows = self.db_connection.execute(
                f"SELECT timestamp, {', '.join(variable_names)} FROM {table} WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
                (str(start.tz_convert("UTC")), str(end.tz_convert("UTC")))
            ).fetchall()

        df = pd.DataFrame(rows, columns=["timestamp"] + variable_names)

        df["timestamp"] = pd.to_datetime(df["timestamp"]).apply(lambda x: x.tz_convert(start.tzinfo if start.tzinfo else "UTC"))
        # Identify missing hours and/or data
        full_range = pd.date_range(start=start, end=end, freq="h", inclusive="left")
        missing = full_range.difference(df["timestamp"]).union(df[df[variable_names].isna().any(axis=1)]["timestamp"])


        # Auto-fetch missing hours
        if len(missing) > 0:
            print(f"Missing {len(missing)} hours between {start} and {end} for ({latitude}, {longitude}). Fetching from Open-Meteo...")
            if len(missing) <2:
                print(f"Missing timestamps:\n{missing}")
            m_start = missing.min()
            m_end = (missing.max() + pd.Timedelta(hours=1))

            self.fetch_openmeteo(latitude, longitude, m_start, m_end, variable_names)
            return self.load_hourly(latitude, longitude, start, end, variable_names)

        return df
    
    # ---------------- Close connection ----------------
    def close(self):
        self.db_connection.close()

def example_usage():
    print("Example usage of MeteoDB class")
    print(f"""
            # Example usage
            db = MeteoDB("./meteo_demo.db")

            # Example coordinates (Montreal)
            latitude = 45.5019
            longitude = -73.5674
            start_Date = pd.Timestamp("2025-09-27T00:00:00").tz_localize("America/Montreal")
            end_Date = pd.Timestamp("2025-10-06T00:00:00").tz_localize("America/Montreal")
            variable_list = ["temperature_2m", "precipitation", "relative_humidity_2m", "weather_code"]

            # Fetch and load data
            df = db.load_hourly(latitude, longitude, start_Date, end_Date, variable_list)
            print(df.head())

            db.close()
          """)

    # Example usage
    db = MeteoDB("./meteo_demo.db")

    # Example coordinates (Montreal)
    latitude = 45.5019
    longitude = -73.5674
    start_Date = pd.Timestamp("2025-10-04T00:00:00").tz_localize("America/Montreal")
    end_Date = pd.Timestamp("2025-10-06T00:00:00").tz_localize("America/Montreal")
    variable_list = ["temperature_2m", "precipitation", "relative_humidity_2m", "weather_code"]

    # Fetch and load data
    df = db.load_hourly(latitude, longitude, start_Date, end_Date, variable_list)
    print(df.head())

    db.close()

if __name__ == "__main__":
    example_usage()