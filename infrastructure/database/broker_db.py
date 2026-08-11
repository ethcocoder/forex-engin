import os
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()

class BrokerConfigModel(Base):
    __tablename__ = 'broker_configs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    broker_id = Column(String(50), nullable=False)
    api_key = Column(String(255), nullable=True)
    account_id = Column(String(100), nullable=True)
    leverage = Column(Float, default=10.0)
    mode = Column(String(20), default='PAPER')
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SystemTelemetryModel(Base):
    __tablename__ = 'system_telemetry'
    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(String(255), nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow)

def get_db_engine(db_path="sqlite:///forex_engin_state.db"):
    engine = create_engine(db_path, echo=False)
    Base.metadata.create_all(engine)
    return engine

def get_session(engine=None):
    if not engine:
        engine = get_db_engine()
    Session = sessionmaker(bind=engine)
    return Session()

if __name__ == "__main__":
    eng = get_db_engine()
    session = get_session(eng)
    print("SQLAlchemy database initialized successfully at forex_engin_state.db")
