import unittest
import time
import datetime
from sqlalchemy import select
from infrastructure.database.timescaledb.connection import TimescaleDBManager
from infrastructure.database.timescaledb.models import Base, Tick, OHLCV, Feature, Order, Trade


class TestTimescaleDBLayer(unittest.TestCase):
    def setUp(self) -> None:
        # Instantiate connection manager
        self.db_manager = TimescaleDBManager()
        
        # Override PostgreSQL engine with a high-speed SQLite in-memory database for isolated unit tests
        from sqlalchemy import create_engine
        from sqlalchemy.orm import scoped_session, sessionmaker
        
        self.db_manager.engine = create_engine("sqlite:///:memory:", echo=False)
        self.db_manager.session_factory = scoped_session(
            sessionmaker(
                bind=self.db_manager.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
        )
        
        # Initialize schema tables
        self.db_manager.create_schema()
        
        # Mock data list
        self.base_time = datetime.datetime.now(datetime.timezone.utc)
        self.test_ticks = [
            {
                "time": self.base_time + datetime.timedelta(milliseconds=i * 100),
                "pair": "EURUSD",
                "bid": 1.1000 + i * 0.0001,
                "ask": 1.1002 + i * 0.0001,
                "volume": 100.0 + i
            }
            for i in range(1000)
        ]

    def tearDown(self) -> None:
        self.db_manager.drop_schema()
        self.db_manager.session_factory.remove()

    def test_schema_creation_and_mappings(self) -> None:
        """Verify that declarative schemas and table mappings are correct."""
        from sqlalchemy import inspect
        inspector = inspect(self.db_manager.engine)
        table_names = inspector.get_table_names()
        
        self.assertIn("ticks", table_names)
        self.assertIn("ohlcv", table_names)
        self.assertIn("features", table_names)
        self.assertIn("orders", table_names)
        self.assertIn("trades", table_names)

    def test_session_scope_lifecycle(self) -> None:
        """Verify that session_scope auto-commits transactions and releases connections correctly."""
        # Write record using session_scope context
        with self.db_manager.session_scope() as session:
            tick = Tick(
                time=self.base_time,
                pair="GBPUSD",
                bid=1.3000,
                ask=1.3002,
                volume=150.0
            )
            session.add(tick)
            
        # Verify the record is persisted and visible in a new session
        with self.db_manager.session_scope() as session:
            stmt = select(Tick).where(Tick.pair == "GBPUSD")
            retrieved = session.execute(stmt).scalar_one()
            self.assertEqual(retrieved.pair, "GBPUSD")
            self.assertAlmostEqual(retrieved.bid, 1.3000)
            self.assertAlmostEqual(retrieved.volume, 150.0)

    def test_bulk_insert_vs_orm_add_performance(self) -> None:
        """
        Verify that SQLAlchemy Core bulk_insert bypasses ORM unit-of-work
        state tracking overhead and is significantly faster than standard ORM additions.
        """
        n_records = len(self.test_ticks)
        
        # 1. Measure standard ORM individual add + commit speed
        with self.db_manager.session_scope() as session:
            start_orm = time.perf_counter()
            for record in self.test_ticks:
                # Need to map dict to model instance for individual add
                tick_instance = Tick(**record)
                session.add(tick_instance)
            session.commit()
            end_orm = time.perf_counter()
            
        orm_duration = end_orm - start_orm
        
        # Verify all records inserted
        with self.db_manager.session_scope() as session:
            count = session.query(Tick).count()
            self.assertEqual(count, n_records)
            
        # Clean up database tables for bulk insert run
        self.db_manager.drop_schema()
        self.db_manager.create_schema()
        
        # 2. Measure Core bulk_insert speed
        with self.db_manager.session_scope() as session:
            start_bulk = time.perf_counter()
            self.db_manager.bulk_insert(session, Tick, self.test_ticks)
            session.commit()
            end_bulk = time.perf_counter()
            
        bulk_duration = end_bulk - start_bulk
        
        # Verify all records inserted
        with self.db_manager.session_scope() as session:
            count = session.query(Tick).count()
            self.assertEqual(count, n_records)
            
        # Print profiling results
        print(f"\nDatabase Ingestion Profile (n={n_records} ticks):")
        print(f"  - Standard ORM Add + Commit: {orm_duration * 1000:.2f} ms")
        print(f"  - SQLAlchemy Core bulk_insert: {bulk_duration * 1000:.2f} ms")
        speedup = orm_duration / bulk_duration if bulk_duration > 0 else 1.0
        print(f"  - Ingestion Speedup Factor: {speedup:.1f}x")
        
        # Assert bulk insert is faster than standard ORM add
        self.assertLess(bulk_duration, orm_duration, "Bulk insert should take less time than ORM add loops")


if __name__ == "__main__":
    unittest.main()
