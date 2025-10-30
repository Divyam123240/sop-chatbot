# backend/database.py

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# ----------------------------
# PostgreSQL Database (Render)
# ----------------------------
DB_USER = "dev_admin"
DB_PASSWORD = "3Q1TxsS4Aa4Msnmrh3RRPE5MnxKUSanZ"
DB_HOST = "dpg-d3d6ica4d50c73d4d12g-a.singapore-postgres.render.com"
DB_PORT = "5432"
DB_NAME = "sop_dev_db"

# SSL mode is required for Render
DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"
)

# ----------------------------
# SQLAlchemy Engine Setup
# ----------------------------
try:
    engine = create_engine(DATABASE_URL, echo=True, pool_pre_ping=True)
except Exception as e:
    raise Exception(f"❌ Failed to create database engine: {e}")

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for ORM models
Base = declarative_base()


# ----------------------------
# Dependency for FastAPI routes
# ----------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
