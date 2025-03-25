import sqlite3

def add_admin_column():
    conn = sqlite3.connect('data/predictions_history.db')
    cursor = conn.cursor()
    
    # ✅ Add is_admin column (if not exists)
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")
        conn.commit()
        print("✅ is_admin column added successfully!")
    except sqlite3.OperationalError:
        print("✅ is_admin column already exists.")

    conn.close()

if __name__ == "__main__":
    add_admin_column()
