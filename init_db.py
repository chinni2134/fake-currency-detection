import sqlite3
import bcrypt # Make sure you have installed this: pip install bcrypt
import getpass # To securely get password input without showing it on the terminal
import os

# Define the name for your database file
DATABASE_NAME = 'users.db' 

def create_user_table():
    """Creates the users table in the database if it doesn't already exist."""
    conn = None  # Initialize conn to None
    try:
        # This will create the database file if it doesn't exist
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT UNIQUE, 
            role TEXT DEFAULT 'analyst' 
        );
        """)
        conn.commit()
        print(f"User table checked/created successfully in '{os.path.abspath(DATABASE_NAME)}'.")
    except sqlite3.Error as e:
        print(f"SQLite error during table creation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during table creation: {e}")
    finally:
        if conn:
            conn.close()

def add_user(username, plain_password, email=None, role='analyst'):
    """Adds a new user to the database with a hashed password."""
    conn = None  # Initialize conn to None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        # Hash the password using bcrypt
        # Ensure plain_password is encoded to bytes
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(plain_password.encode('utf-8'), salt)

        cursor.execute("""
        INSERT INTO users (username, password_hash, email, role)
        VALUES (?, ?, ?, ?)
        """, (username, hashed_password.decode('utf-8'), email, role)) # Store hash as string
        conn.commit()
        print(f"User '{username}' added successfully!")
    except sqlite3.IntegrityError:
        # This error occurs if the username or email already exists (due to UNIQUE constraint)
        print(f"Error: Username '{username}' or Email (if provided) already exists in the database.")
    except sqlite3.Error as e:
        print(f"SQLite error during user insertion: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during user insertion: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    # Step 1: Ensure the database and table structure is ready
    create_user_table()

    print("\n--- Create Initial User for Fake Currency Detection App ---")
    
    try:
        # Step 2: Get details for the initial user
        initial_username = input("Enter username for the initial user: ").strip()
        
        # Using getpass to hide password input for better security on the command line
        initial_password = getpass.getpass(f"Enter password for '{initial_username}' (input will be hidden): ").strip()
        confirm_password = getpass.getpass(f"Confirm password for '{initial_username}': ").strip()

        if initial_password != confirm_password:
            print("Passwords do not match. User creation aborted.")
        elif not initial_username or not initial_password:
            print("Username and password cannot be empty. User creation aborted.")
        else:
            initial_email = input(f"Enter email for '{initial_username}' (optional, press Enter to skip): ").strip()
            # If email is empty, store None (or NULL in SQL)
            initial_email = initial_email if initial_email else None 
            
            # Step 3: Add the user to the database
            add_user(initial_username, initial_password, initial_email)
            print(f"\nDatabase '{DATABASE_NAME}' is ready and user '{initial_username}' has been added (if not already existing).")
            print(f"You can now run your main Flask application (app.py) and try logging in.")

    except Exception as e:
        print(f"An error occurred during the user input process: {e}")
