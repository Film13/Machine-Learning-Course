from pathlib import Path

# กำหนด Path ของไฟล์ข้อมูล
DATA_PATH = Path('cats.csv')
if not DATA_PATH.exists():
    DATA_PATH = Path('LAB 5') / 'cats.csv'

if not DATA_PATH.exists():
    raise FileNotFoundError(f'Cannot find dataset: {DATA_PATH.resolve()}')