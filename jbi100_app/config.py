from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

SERVICES_PATH = DATA_DIR / "services_weekly.csv"
SCHEDULE_PATH = DATA_DIR / "staff_schedule.csv"
PATIENTS_PATH = DATA_DIR / "patients.csv"