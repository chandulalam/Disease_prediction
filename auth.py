import sqlite3
import bcrypt
import streamlit as st

# ✅ Create User Table (with is_admin column)
def create_user_table():
    conn = sqlite3.connect('data/predictions_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

# ✅ Signup Function
def signup(username, password):
    conn = sqlite3.connect('data/predictions_history.db')
    cursor = conn.cursor()

    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
        st.success("Signup successful! Please log in.")
    except sqlite3.IntegrityError:
        st.error("Username already exists. Try another one.")

    conn.close()

# ✅ Login Function
def login(username, password):
    conn = sqlite3.connect('data/predictions_history.db')
    cursor = conn.cursor()

    cursor.execute("SELECT password, is_admin FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()

    conn.close()

    if result:
        hashed_pw, is_admin = result
        if bcrypt.checkpw(password.encode('utf-8'), hashed_pw):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.session_state['is_admin'] = bool(is_admin)  # ✅ Store admin status
            st.success(f"Welcome, {username}! 🎉")
            return True, bool(is_admin)   # ✅ Return admin status
        else:
            st.error("Incorrect password.")
            return False, False
    else:
        st.error("User not found.")
        return False, False


# ✅ Logout Function
def logout():
    st.session_state['logged_in'] = False
    st.session_state['username'] = None
    st.session_state['is_admin'] = False
    st.success("Logged out successfully.")

# ✅ Check Admin Status
def is_admin(username):
    conn = sqlite3.connect('data/predictions_history.db')
    cursor = conn.cursor()
    cursor.execute("SELECT is_admin FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    return result and result[0] == 1

# ✅ Promote User to Admin
def set_admin(username):
    conn = sqlite3.connect('data/predictions_history.db')
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE users SET is_admin = 1 WHERE username = ?", (username,))
        conn.commit()
        st.success(f"{username} has been promoted to Admin! ✅")

        # ✅ If the currently logged-in user was promoted, update session
        if username == st.session_state['username']:
            st.session_state['is_admin'] = True
            st.rerun()  # Refresh the page
        return True
    except Exception as e:
        st.error(f"Failed to promote {username}: {e}")
        return False
    finally:
        conn.close()

# ✅ Create table on startup
create_user_table()
