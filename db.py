"""Database layer.

Single DATABASE_URL env var picks the backend:
- unset / sqlite://...  -> SQLite, embeddings stored as BLOB, similarity in numpy
- postgresql://...      -> Postgres; if pgvector extension is present, embeddings are
                           stored as vector(512) and similarity uses the <=> operator
                           server-side. Otherwise falls back to bytea + numpy.

Same models, same code paths everywhere — the only thing that changes is where the
nearest-neighbor lookup happens.
"""
from __future__ import annotations

import logging
import os
from datetime import date as Date, datetime, time as Time, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
from sqlalchemy import (
    Column, Integer, String, Float, LargeBinary, Date as DateCol, DateTime, Time as TimeCol,
    ForeignKey, UniqueConstraint, create_engine, event, select, func, text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

log = logging.getLogger("attendance.db")

DEFAULT_SQLITE_URL = "sqlite:///attendance.db"
DATABASE_URL = os.environ.get("DATABASE_URL") or DEFAULT_SQLITE_URL

IS_POSTGRES = DATABASE_URL.startswith(("postgresql://", "postgres://"))
# Normalize: SQLAlchemy 2.x wants postgresql:// not postgres://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://"):]

PGVECTOR_AVAILABLE = False  # decided at init time

Base = declarative_base()


# ---------- Models ----------------------------------------------------------

class Person(Base):
    __tablename__ = "persons"
    id = Column(Integer, primary_key=True)
    external_id = Column(String(64), unique=True, nullable=False)
    name = Column(String(128), nullable=False)
    group_name = Column(String(64), nullable=True)
    role = Column(String(16), nullable=False, default="student")  # 'student' | 'worker'
    embedding = Column(LargeBinary, nullable=True)  # may be replaced with vector(512) at runtime
    embedding_dim = Column(Integer, nullable=False, default=512)
    thumbnail_path = Column(String(256), nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)


class Session_(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    name = Column(String(128), nullable=False)
    mode = Column(String(16), nullable=False, default="student")  # 'student' | 'worker'
    group_name = Column(String(64), nullable=True)
    start_time = Column(TimeCol, nullable=False, default=Time(9, 0))
    late_threshold_minutes = Column(Integer, nullable=False, default=15)
    duration_minutes = Column(Integer, nullable=True)  # None = manual end only
    date = Column(DateCol, nullable=False, default=Date.today)
    __table_args__ = (UniqueConstraint("name", "date", name="uq_session_name_date"),)


class Attendance(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    arrival_ts = Column(DateTime, nullable=False, default=datetime.now)
    departure_ts = Column(DateTime, nullable=True)
    status = Column(String(16), nullable=False, default="present")  # present|late|absent
    person = relationship("Person")
    session = relationship("Session_")
    __table_args__ = (UniqueConstraint("person_id", "session_id", name="uq_attendance_person_session"),)


class Unknown(Base):
    __tablename__ = "unknowns"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True)
    seen_ts = Column(DateTime, nullable=False, default=datetime.now)
    thumbnail_path = Column(String(256), nullable=True)


class FaceSample(Base):
    """Multiple embeddings per person — front/left/right/etc.
    Recognition matches against the MIN distance across all of a person's samples,
    so a single match against any pose is enough to identify them."""
    __tablename__ = "face_samples"
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False)
    embedding = Column(LargeBinary, nullable=True)  # may be replaced with vector(512) at runtime
    embedding_dim = Column(Integer, nullable=False, default=512)
    pose_label = Column(String(32), nullable=True)  # 'front' | 'left' | 'right' | None
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    person = relationship("Person")


class SessionMember(Base):
    __tablename__ = "session_members"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False)
    session = relationship("Session_")
    person = relationship("Person")
    __table_args__ = (UniqueConstraint("session_id", "person_id", name="uq_session_member"),)


class Config(Base):
    __tablename__ = "config"
    key = Column(String(64), primary_key=True)
    value = Column(String(512), nullable=False)


# ---------- Engine / init ---------------------------------------------------

def _make_engine():
    kwargs = {"future": True}
    if IS_POSTGRES:
        kwargs["pool_pre_ping"] = True  # Neon idle-connection drops
    else:
        # SQLite: allow access from Flask request threads
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(DATABASE_URL, **kwargs)


engine = _make_engine()

if not IS_POSTGRES:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _rec):
        dbapi_conn.execute("PRAGMA foreign_keys = ON")

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> dict:
    """Create tables. If on Postgres with pgvector available, swap embedding to vector(512).
    Returns a small status dict for logging.
    """
    global PGVECTOR_AVAILABLE
    info: dict = {"backend": "postgres" if IS_POSTGRES else "sqlite", "pgvector": False, "url_redacted": _redact(DATABASE_URL)}

    if IS_POSTGRES:
        try:
            with engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            PGVECTOR_AVAILABLE = True
            info["pgvector"] = True
        except Exception as e:
            log.warning("pgvector not available: %s — falling back to bytea+numpy", e)
            PGVECTOR_AVAILABLE = False

        if PGVECTOR_AVAILABLE:
            # Replace the embedding column types before create_all so they
            # become native vector(512) columns indexed for cosine similarity.
            from pgvector.sqlalchemy import Vector
            Person.__table__.c.embedding.type = Vector(512)
            FaceSample.__table__.c.embedding.type = Vector(512)

    Base.metadata.create_all(engine)

    # Lightweight migrations for columns added after initial schema
    _migrate_add_column(engine, "sessions", "duration_minutes", "INTEGER")

    return info


def _migrate_add_column(eng, table: str, column: str, col_type: str) -> None:
    """Add a column if it doesn't exist yet. Works for both SQLite and Postgres."""
    try:
        with eng.begin() as conn:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
        log.info("Migrated: added %s.%s", table, column)
    except Exception:
        pass  # column already exists


def _redact(url: str) -> str:
    if "@" not in url:
        return url
    head, tail = url.split("@", 1)
    if "://" in head and ":" in head.split("://", 1)[1]:
        scheme, rest = head.split("://", 1)
        user = rest.split(":", 1)[0]
        return f"{scheme}://{user}:***@{tail}"
    return url


# ---------- Embedding helpers ----------------------------------------------

def encode_embedding(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def decode_embedding(blob: bytes | list | np.ndarray) -> np.ndarray:
    if isinstance(blob, np.ndarray):
        return blob.astype(np.float32)
    if isinstance(blob, list):
        return np.asarray(blob, dtype=np.float32)
    return np.frombuffer(blob, dtype=np.float32)


def find_nearest(db: Session, query: np.ndarray, threshold: float = 0.5) -> tuple[Person | None, float]:
    """Return (person, cosine_distance) or (None, distance) if above threshold.

    Strategy: each person can have multiple face_samples (front/left/right/etc).
    The match score for a person is the MINIMUM distance across all their samples.
    So a single matching pose is enough to identify them. Falls back to the legacy
    Person.embedding column if a person has no samples yet (older enrolled rows).
    """
    q = np.asarray(query, dtype=np.float32)
    qn = q / (np.linalg.norm(q) + 1e-9)

    if IS_POSTGRES and PGVECTOR_AVAILABLE:
        # Server-side aggregate: closest sample per person, then global argmin.
        row = db.execute(
            text(
                "SELECT person_id AS id, MIN(embedding <=> CAST(:q AS vector)) AS dist "
                "FROM face_samples WHERE embedding IS NOT NULL "
                "GROUP BY person_id ORDER BY dist ASC LIMIT 1"
            ),
            {"q": qn.tolist()},
        ).first()
        if row is None:
            # Legacy: persons enrolled before face_samples existed.
            row = db.execute(
                text(
                    "SELECT id, embedding <=> CAST(:q AS vector) AS dist "
                    "FROM persons WHERE embedding IS NOT NULL ORDER BY dist ASC LIMIT 1"
                ),
                {"q": qn.tolist()},
            ).first()
        if row is None:
            return None, 1.0
        person = db.get(Person, row.id)
        return (person if row.dist <= threshold else None), float(row.dist)

    # SQLite path: numpy comparison, min over samples per person.
    qdim = q.shape[0]
    samples = db.execute(
        select(FaceSample, Person).join(Person, FaceSample.person_id == Person.id)
        .where(FaceSample.embedding.is_not(None))
    ).all()
    person_best: dict[int, tuple[Person, float]] = {}
    for fs, p in samples:
        e = decode_embedding(fs.embedding)
        if e.shape[0] != qdim:
            continue
        en = e / (np.linalg.norm(e) + 1e-9)
        d = float(1.0 - np.dot(qn, en))
        if p.id not in person_best or d < person_best[p.id][1]:
            person_best[p.id] = (p, d)

    if not person_best:
        legacy = db.execute(select(Person).where(Person.embedding.is_not(None))).scalars().all()
        for p in legacy:
            e = decode_embedding(p.embedding)
            if e.shape[0] != qdim:
                continue
            en = e / (np.linalg.norm(e) + 1e-9)
            d = float(1.0 - np.dot(qn, en))
            if p.id not in person_best or d < person_best[p.id][1]:
                person_best[p.id] = (p, d)

    if not person_best:
        return None, 1.0
    ranked = sorted(person_best.values(), key=lambda x: x[1])
    best_p, best_d = ranked[0]
    if best_d > threshold:
        return None, best_d
    if len(ranked) >= 2:
        second_d = ranked[1][1]
        margin = second_d - best_d
        if margin < 0.05:
            log.debug("Ambiguous match: best=%.3f second=%.3f margin=%.3f — rejecting", best_d, second_d, margin)
            return None, best_d
    return best_p, best_d


def add_face_sample(db: Session, person: Person, vec: np.ndarray, pose_label: str | None = None) -> None:
    """Append a new pose sample for `person`. Stores native vector(512) on
    Postgres+pgvector, raw float32 BLOB on SQLite."""
    v = np.asarray(vec, dtype=np.float32)
    sample = FaceSample(person_id=person.id, embedding_dim=int(v.shape[0]), pose_label=pose_label)
    if IS_POSTGRES and PGVECTOR_AVAILABLE:
        sample.embedding = v.tolist()
    else:
        sample.embedding = encode_embedding(v)
    db.add(sample)


def count_samples(db: Session, person_id: int) -> int:
    return int(db.execute(
        select(func.count(FaceSample.id)).where(FaceSample.person_id == person_id)
    ).scalar() or 0)


def store_embedding(person: Person, vec: np.ndarray) -> None:
    """Set the embedding correctly for the active backend."""
    v = np.asarray(vec, dtype=np.float32)
    if IS_POSTGRES and PGVECTOR_AVAILABLE:
        person.embedding = v.tolist()  # pgvector adapter handles list -> vector
    else:
        person.embedding = encode_embedding(v)
    person.embedding_dim = int(v.shape[0])


# ---------- Config (camera selection persistence) --------------------------

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


# ---------- Session members ------------------------------------------------

def set_session_members(db: Session, session_id: int, person_ids: list[int]) -> None:
    for old in db.execute(
        select(SessionMember).where(SessionMember.session_id == session_id)
    ).scalars().all():
        db.delete(old)
    db.flush()
    for pid in person_ids:
        db.add(SessionMember(session_id=session_id, person_id=pid))
    db.commit()


def get_session_member_ids(db: Session, session_id: int) -> list[int]:
    rows = db.execute(
        select(SessionMember.person_id).where(SessionMember.session_id == session_id)
    ).scalars().all()
    return list(rows)


# ---------- Attendance helpers ---------------------------------------------

def ensure_session(db: Session, name: str, mode: str, group: str | None,
                   start_time: Time | None = None, late_minutes: int = 15,
                   for_date: Date | None = None,
                   duration_minutes: int | None = None) -> Session_:
    """Idempotent on (name, date). If a session already exists for the given
    date, returns it; otherwise creates one. Defaults to today's date."""
    d = for_date or Date.today()
    s = db.execute(
        select(Session_).where(Session_.name == name, Session_.date == d)
    ).scalar_one_or_none()
    if s:
        return s
    s = Session_(
        name=name, mode=mode, group_name=group, date=d,
        start_time=start_time or Time(9, 0), late_threshold_minutes=late_minutes,
        duration_minutes=duration_minutes,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


# Back-compat alias
ensure_today_session = ensure_session


def record_arrival(db: Session, person_id: int, session: Session_) -> Attendance | None:
    """Insert the first arrival for (person, session). Returns the new row, or None
    if one already existed (this is the dedupe path)."""
    existing = db.execute(
        select(Attendance).where(
            Attendance.person_id == person_id, Attendance.session_id == session.id
        )
    ).scalar_one_or_none()
    if existing:
        # update departure to "now" for worker mode (clock-out tracking)
        existing.departure_ts = datetime.now()
        db.commit()
        return None

    now = datetime.now()
    status = "present"
    if session.mode == "student":
        cutoff = datetime.combine(session.date, session.start_time) + timedelta(
            minutes=session.late_threshold_minutes
        )
        if now > cutoff:
            status = "late"
    a = Attendance(person_id=person_id, session_id=session.id, arrival_ts=now, status=status)
    db.add(a)
    db.commit()
    db.refresh(a)
    return a
