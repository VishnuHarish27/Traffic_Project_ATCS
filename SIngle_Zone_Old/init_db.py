import sqlite3

def init_db(database):
    """
    Initializes the SQLite database and creates the traffic_data table if it doesn't exist.
    """
    try:
        conn = sqlite3.connect(database)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS traffic_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                vehicle_count INTEGER,
                density_status TEXT
            )
        ''')
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"Error initializing database: {e}")
