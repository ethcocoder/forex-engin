import contextlib
import structlog
from typing import Any, Dict, Generator, List, Type
from sqlalchemy import create_engine, insert
from sqlalchemy.orm import scoped_session, sessionmaker, Session
from infrastructure.database.timescaledb.models import Base

logger = structlog.get_logger()


class TimescaleDBManager:
    """
    Database Connection Pool & Session Manager for TimescaleDB.
    Implements production-grade pooling and thread-safe session management.
    """

    def __init__(self, config: Any = None) -> None:
        self.config = config
        
        # Load connection properties with robust local defaults
        if config and hasattr(config, "database") and hasattr(config.database, "timescaledb"):
            ts_cfg = config.database.timescaledb
            host = ts_cfg.host
            port = ts_cfg.port
            username = ts_cfg.username
            password = ts_cfg.password
            database = ts_cfg.database
        else:
            host = "localhost"
            port = 5432
            username = "postgres"
            password = "password"
            database = "forex_db"
            
        self.connection_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
        
        # Initialize SQLAlchemy engine with production-grade pooling
        logger.info(
            "Initializing TimescaleDB connection pool",
            host=host,
            port=port,
            database=database,
            pool_size=10,
            max_overflow=20
        )
        
        self.engine = create_engine(
            self.connection_url,
            pool_size=10,
            max_overflow=20,
            pool_recycle=1800,
            pool_pre_ping=True,
            echo=False
        )
        
        # Scoped thread-safe session factory
        self.session_factory = scoped_session(
            sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
        )

    def create_schema(self) -> None:
        """Creates database schema tables if they do not exist (useful for testing)."""
        logger.info("Initializing TimescaleDB schemas")
        Base.metadata.create_all(self.engine)

    def drop_schema(self) -> None:
        """Drops all database schema tables (useful for cleanup in tests)."""
        logger.info("Dropping TimescaleDB schemas")
        Base.metadata.drop_all(self.engine)

    @contextlib.contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Context manager for thread-safe database transactional sessions.
        Automatically commits operations or rolls back on exception, then closes session.
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Database transaction failed, rolling back", error=str(e))
            raise e
        finally:
            session.close()

    def bulk_insert(self, session: Session, model: Type[Base], data_list: List[Dict[str, Any]]) -> None:
        """
        Bypasses standard SQLAlchemy ORM unit-of-work state tracking overhead by
        executing a vectorized SQLAlchemy Core batch insert statement.
        This provides maximal write performance matching raw SQL copy/insert.
        """
        if not data_list:
            return
            
        try:
            # Vectorized multi-row insert using SQLAlchemy Core
            session.execute(insert(model), data_list)
        except Exception as e:
            logger.error("Bulk insert failed", model=model.__name__, error=str(e))
            raise e
