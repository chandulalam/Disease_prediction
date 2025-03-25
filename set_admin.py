import sqlite3

def make_admin(username):
    conn = sqlite3.connect('data/predictions_history.db')
    cursor = conn.cursor()
    
    # Update is_admin field to 1
    cursor.execute("UPDATE users SET is_admin = 1 WHERE username = ?", (username,))
    conn.commit()
    
    print(f"✅ User '{username}' is now an admin!")
    conn.close()

if __name__ == "__main__":
    username = input("Enter the username to make admin: ")
    make_admin(username)
