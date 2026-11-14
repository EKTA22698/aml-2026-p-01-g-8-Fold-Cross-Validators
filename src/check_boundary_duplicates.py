
from pathlib import Path
import sqlite3
import tempfile
import pandas as pd

DATA_DIR = Path("data/citibike_2026")
CHUNK_SIZE = 100_000

MONTHS = {"june", "july", "august", "september", "october"}

files = sorted(
    f for f in DATA_DIR.rglob("*.csv")
    if f.parent.name.lower() in MONTHS
)

if not files:
    raise FileNotFoundError("No monthly CSV files found.")

total_rows = 0
missing_ids = 0
duplicate_rows = 0
unique_ids = 0

print(f"CSV files found: {len(files)}")
print("Starting global ride_id duplicate audit...\n")

# Temporary database: IDs are stored on disk, not in a Python set.
with tempfile.TemporaryDirectory(prefix="citibike_audit_") as temp_dir:
    db_path = Path(temp_dir) / "ride_ids.sqlite"

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("""
            CREATE TABLE ride_ids (
                ride_id TEXT PRIMARY KEY
            )
        """)

        for file_number, file in enumerate(files, start=1):
            file_rows = 0

            print(f"[{file_number}/{len(files)}] Scanning {file.name}")

            for chunk in pd.read_csv(
                file,
                usecols=["ride_id"],
                dtype={"ride_id": "string"},
                chunksize=CHUNK_SIZE
            ):
                total_rows += len(chunk)
                file_rows += len(chunk)

                ids = chunk["ride_id"].dropna().str.strip()
                ids = ids[ids != ""]

                missing_ids += len(chunk) - len(ids)

                before = conn.total_changes

                conn.executemany(
                    "INSERT OR IGNORE INTO ride_ids (ride_id) VALUES (?)",
                    ((ride_id,) for ride_id in ids)
                )

                inserted = conn.total_changes - before

                unique_ids += inserted
                duplicate_rows += len(ids) - inserted

            print(f"    Rows scanned: {file_rows:,}")

        conn.commit()

print("\n========== GLOBAL DUPLICATE AUDIT ==========")
print(f"Total rows scanned: {total_rows:,}")
print(f"Valid ride IDs: {total_rows - missing_ids:,}")
print(f"Unique ride IDs: {unique_ids:,}")
print(f"Duplicate rows: {duplicate_rows:,}")
print(f"Missing/blank ride IDs: {missing_ids:,}")
print("Audit completed.")
