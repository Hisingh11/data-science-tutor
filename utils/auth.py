import hashlib
from utils.database import SessionLocal, User
from datetime import datetime

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username: str, password: str, full_name: str = ""):
    db = SessionLocal()
    existing = db.query(User).filter(User.username == username).first()
    if existing:
        db.close()
        return False, "Username exists"
    user = User(
        username=username,
        password_hash=hash_password(password),
        full_name=full_name,
        created_at=datetime.utcnow()
    )
    db.add(user)
    db.commit()
    db.close()
    return True, "Registered"

def login_user(username: str, password: str):
    db = SessionLocal()
    user = db.query(User).filter(User.username == username).first()
    db.close()
    if user and user.password_hash == hash_password(password):
        return user.id, user.username, user.full_name
    return None