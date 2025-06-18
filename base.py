"""
Database Base Configuration
SQLAlchemy base class and metadata
"""

from sqlalchemy.orm import DeclarativeBase, declared_attr
from sqlalchemy import MetaData

# Naming convention for constraints
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Base(DeclarativeBase):
    """Base class for all database models"""
    metadata = metadata
    
    @declared_attr
    def __tablename__(cls) -> str:
        """Generate table name from class name"""
        return cls.__name__.lower()
    
    def dict(self):
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


# Import all models to ensure they are registered with Base
from app.db.models.user import User  # noqa
from app.db.models.patient import Patient  # noqa
from app.db.models.ecg_analysis import ECGAnalysis  # noqa
from app.db.models.validation import Validation  # noqa
from app.db.models.notification import Notification  # noqa

__all__ = ["Base", "metadata"]
