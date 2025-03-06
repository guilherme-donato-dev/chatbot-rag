import sqlite3
import os

print("SQLite Version (python):", sqlite3.sqlite_version)
print("SQLite library path:", os.path.realpath(sqlite3.__file__))
