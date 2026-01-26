import sqlite3
import os

# Veritabanı dosyası için sabit değişken
DB_NAME = "students.db"

def initialize_database():
    """
    Mevcut veritabanını temizler ve yeni bir bağlantı oluşturur.
    Portfolyo gösterimi için her çalıştırışta sıfırdan başlar.
    """
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    return conn, cursor

def create_tables(cursor):
    """
    Öğrenci ve Kurs tablolarını oluşturur (DDL).
    """
    # Students Tablosu
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Students (
            id INTEGER PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            age INTEGER,
            email VARCHAR(255) UNIQUE,
            city VARCHAR(255)
        )
    ''')
    
    # Courses Tablosu
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Courses (
            id INTEGER PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            instructor VARCHAR(255),
            credits INTEGER
        )
    ''')

def insert_sample_data(cursor):
    """
    Tablolara örnek veriler ekler (DML).
    """
    students = [
        (1, 'Alice Johnson', 20, 'alice@gmail.com', 'New York'),
        (2, 'Bob Smith', 19, 'bob@gmail.com', 'Chicago'),
        (3, 'Carol White', 21, 'carol@gmail.com', 'Boston'),
        (4, 'David Brown', 20, 'david@gmail.com', 'New York'),
        (5, 'Emma Davis', 21, 'emma@gmail.com', 'Seattle')        
    ]
    
    # executemany performans açısından toplu veri eklemede tercih edilir
    cursor.executemany("INSERT INTO Students VALUES (?,?,?,?,?)", students)
    print(f"{len(students)} records inserted into Students table.")

def fetch_and_display_data(cursor):
    """
    Verileri sorgular ve ekrana yazdırır (DQL).
    """
    print("\n--- Current Student List ---")
    cursor.execute("SELECT * FROM Students")
    records = cursor.fetchall()
    
    # Verileri daha okunaklı yazdırma
    print(f"{'ID':<5} {'Name':<20} {'Age':<5} {'City':<15} {'Email'}")
    print("-" * 60)
    for row in records:
        print(f"{row[0]:<5} {row[1]:<20} {row[2]:<5} {row[4]:<15} {row[3]}")

def main():
    conn, cursor = initialize_database()
    
    try:
        create_tables(cursor)
        insert_sample_data(cursor)
        conn.commit() # Değişiklikleri kaydet
        
        fetch_and_display_data(cursor)
        
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        
    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

if __name__ == "__main__":
    main()