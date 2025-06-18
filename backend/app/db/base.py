"""Database base configuration."""
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

Base = declarative_base()

class DatabaseBase:
    """Base class for database models."""
    __abstract__ = True

# For compatibility
metadata = Base.metadata
