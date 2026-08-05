import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.psx_predictor.db.connection import engine
from src.psx_predictor.db.models import Base

if __name__ == "__main__":
    print("Creating all tables...")
    Base.metadata.create_all(engine)
    print("Tables created successfully.")
