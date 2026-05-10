"""Database layer. We built this to handle all data persistence for the attendance system."""
import logging
import os
from datetime import date as Date, datetime, time as Time, timedelta
from sqlalchemy import Column, Integer, String, DateTime, Time as TimeCol, Date as DateCol, ForeignKey, UniqueConstraint, create_engine, event, select
from sqlalchemy.orm import declarative_base, sessionmaker, Session

log = logging.getLogger("attendance.db")

# I set up the engine to support both SQLite and Postgres seamlessly.
DATABASE_URL = os.environ.get("DATABASE_URL") or "sqlite:///attendance.db"
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://"):]

IS_POSTGRES = DATABASE_URL.startswith("postgresql://")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if not IS_POSTGRES else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

if not IS_POSTGRES:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _rec):
        dbapi_conn.execute("PRAGMA foreign_keys = ON")

Base = declarative_base()

class Person(Base):
    __tablename__ = "persons"
    id = Column(Integer, primary_key=True)
    external_id = Column(String(64), unique=True, nullable=False)
    name = Column(String(128), nullable=False)
    group_name = Column(String(64), nullable=True)
    role = Column(String(16), nullable=False, default="student")
    created_at = Column(DateTime, default=datetime.now, nullable=False)

class Session_(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    name = Column(String(128), nullable=False)
    mode = Column(String(16), nullable=False, default="student")
    group_name = Column(String(64), nullable=True)
    start_time = Column(TimeCol, nullable=False, default=Time(9, 0))
    late_threshold_minutes = Column(Integer, nullable=False, default=15)
    duration_minutes = Column(Integer, nullable=True)
    date = Column(DateCol, nullable=False, default=Date.today)
    __table_args__ = (UniqueConstraint("name", "date", name="uq_session_name_date"),)

class Attendance(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    arrival_ts = Column(DateTime, nullable=False, default=datetime.now)
    status = Column(String(16), nullable=False, default="present")
    __table_args__ = (UniqueConstraint("person_id", "session_id", name="uq_attendance_person_session"),)

class Config(Base):
    __tablename__ = "config"
    key = Column(String(64), primary_key=True)
    value = Column(String(512), nullable=False)

def init_db():
    I created this function to ensure all tables are created on startup.
    Base.metadata.create_all(engine)
    log.info("Database initialized successfully.")

def get_config(db: Session, key: str, default: str | None = None) -> str | None:
    row = db.get(Config, key)
    return row.value if row else default

def set_config(db: Session, key: str, value: str) -> None:
    row = db.get(Config, key)
    if row:
        row.value = value
    else:
        db.add(Config(key=key, value=value))
    db.commit()

def del_config(db: Session, key: str) -> None:
    row = db.get(Config, key)
    if row:
        db.delete(row)
        db.commit()

def ensure_session(db: Session, name: str, mode: str, group: str | None = None,
                   start_time: Time | None = None, late_minutes: int = 15,
                   for_date: Date | None = None, duration_minutes: int | None = None) -> Session_:
    I wrote this to create sessions idempotently based on name and date.
    d = for_date or Date.today()
    s = db.execute(select(Session_).where(Session_.name == name, Session_.date == d)).scalar_one_or_none()
    if s:
        return s
    s = Session_(name=name, mode=mode, group_name=group, date=d,
                 start_time=start_time or Time(9, 0), late_threshold_minutes=late_minutes,
                 duration_minutes=duration_minutes)
    db.add(s)
    db.commit()
    db.refresh(s)
    return s

def record_arrival(db: Session, person_id: int, session: Session_) -> Attendance | None:
    I implemented this to log the first arrival and handle late/present status automatically.
    existing = db.execute(select(Attendance).where(
        Attendance.person_id == person_id, Attendance.session_id == session.id
    )).scalar_one_or_none()
    if existing:
        return None

    now = datetime.now()
    status = "present"
    if session.mode == "student":
        cutoff = datetime.combine(session.date, session.start_time) + timedelta(minutes=session.late_threshold_minutes)
        if now > cutoff:
            status = "late"

    a = Attendance(person_id=person_id, session_id=session.id, arrival_ts=now, status=status)
    db.add(a)
    db.commit()
    db.refresh(a)
    return a