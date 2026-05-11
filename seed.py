"""Seed the database with fake students, workers, and a few days of attendance.

NOTE: this does NOT create fake face embeddings. Real enrollment must happen via
the webcam at demo time. This is just so the dashboard has data to display.

Usage:
    python seed.py            # add seed data (idempotent on external_id)
    python seed.py --reset    # drop and recreate ALL tables (destructive!)
"""
from __future__ import annotations

import argparse
import logging
import random
from datetime import date, datetime, time, timedelta

from sqlalchemy import select

import db as DB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("seed")

STUDENTS = [
    ("S1001", "Alice Anderson"), ("S1002", "Bilal Bashir"),
    ("S1003", "Carmen Castillo"), ("S1004", "Devi Desai"),
    ("S1005", "Eitan Erlich"),    ("S1006", "Fatima Farouk"),
    ("S1007", "Gabriel Gomez"),   ("S1008", "Hana Haruki"),
]
WORKERS = [
    ("W2001", "Iva Ivanova"),    ("W2002", "Jakub Janik"),
    ("W2003", "Kenji Kobayashi"),("W2004", "Lola Lopez"),
    ("W2005", "Marek Mazur"),
]


def reset_all():
    DB.Base.metadata.drop_all(DB.engine)
    DB.Base.metadata.create_all(DB.engine)
    log.info("All tables dropped and recreated.")


def upsert_person(s, ext_id, name, group, role):
    p = s.execute(select(DB.Person).where(DB.Person.external_id == ext_id)).scalar_one_or_none()
    if p:
        return p
    p = DB.Person(external_id=ext_id, name=name, group_name=group, role=role)
    s.add(p)
    s.flush()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="DROP and recreate all tables")
    args = ap.parse_args()

    DB.init_db()
    if args.reset:
        reset_all()

    rng = random.Random(42)
    with DB.SessionLocal() as s:
        students = [upsert_person(s, ext, n, "CS101", "student") for ext, n in STUDENTS]
        workers = [upsert_person(s, ext, n, "Engineering", "worker") for ext, n in WORKERS]
        s.commit()

        # Fake sessions for the last 5 weekdays
        today = date.today()
        for back in range(5, 0, -1):
            d = today - timedelta(days=back)
            if d.weekday() >= 5:
                continue
            sess = s.execute(
                select(DB.Session_).where(DB.Session_.name == "CS101 morning", DB.Session_.date == d)
            ).scalar_one_or_none()
            if sess is None:
                sess = DB.Session_(name="CS101 morning", mode="student", group_name="CS101",
                                   date=d, start_time=time(9, 0), late_threshold_minutes=15)
                s.add(sess); s.flush()

            for p in students:
                already = s.execute(select(DB.Attendance).where(
                    DB.Attendance.person_id == p.id, DB.Attendance.session_id == sess.id
                )).scalar_one_or_none()
                if already:
                    continue
                roll = rng.random()
                if roll < 0.7:
                    arrival = datetime.combine(d, time(8, 50)) + timedelta(minutes=rng.randint(0, 12))
                    status = "present"
                elif roll < 0.9:
                    arrival = datetime.combine(d, time(9, 16)) + timedelta(minutes=rng.randint(0, 25))
                    status = "late"
                else:
                    continue  # absent
                s.add(DB.Attendance(person_id=p.id, session_id=sess.id,
                                    arrival_ts=arrival, status=status))

            # Worker session for the same day
            wsess = s.execute(
                select(DB.Session_).where(DB.Session_.name == "Engineering shift", DB.Session_.date == d)
            ).scalar_one_or_none()
            if wsess is None:
                wsess = DB.Session_(name="Engineering shift", mode="worker", group_name="Engineering",
                                    date=d, start_time=time(8, 0), late_threshold_minutes=0)
                s.add(wsess); s.flush()

            for p in workers:
                already = s.execute(select(DB.Attendance).where(
                    DB.Attendance.person_id == p.id, DB.Attendance.session_id == wsess.id
                )).scalar_one_or_none()
                if already:
                    continue
                if rng.random() < 0.85:
                    arrival = datetime.combine(d, time(8, 0)) + timedelta(minutes=rng.randint(-15, 30))
                    departure = arrival + timedelta(hours=8, minutes=rng.randint(-20, 30))
                    s.add(DB.Attendance(person_id=p.id, session_id=wsess.id,
                                        arrival_ts=arrival, departure_ts=departure, status="present"))
        s.commit()
        log.info("Seeded %d students, %d workers, plus attendance for the last few days.",
                 len(students), len(workers))


if __name__ == "__main__":
    main()
