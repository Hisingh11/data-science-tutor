import json
from utils.database import SessionLocal, ChatSession, ChatMessage
from datetime import datetime, timezone

def create_chat_session(user_id: int, title: str = "New Chat"):
    db = SessionLocal()
    session = ChatSession(user_id=user_id, title=title)
    db.add(session)
    db.commit()
    session_id = session.id
    db.close()
    return session_id

def add_message(session_id: int, role: str, content: str, attachments: list = None):
    db = SessionLocal()
    attachments_json = json.dumps(attachments or [])
    msg = ChatMessage(session_id=session_id, role=role, content=content, attachments=attachments_json)
    db.add(msg)
    db.query(ChatSession).filter(ChatSession.id == session_id).update({"updated_at": datetime.now(timezone.utc)})
    db.commit()
    db.close()

def get_session_messages(session_id: int):
    db = SessionLocal()
    msgs = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp).all()
    db.close()
    return [{"role": m.role, "content": m.content, "attachments": json.loads(m.attachments) if m.attachments else []} for m in msgs]

def get_user_sessions(user_id: int):
    db = SessionLocal()
    sessions = db.query(ChatSession).filter(ChatSession.user_id == user_id).order_by(ChatSession.updated_at.desc()).all()
    db.close()
    return [{"id": s.id, "title": s.title, "updated_at": s.updated_at.isoformat()} for s in sessions]

def delete_session(session_id: int):
    db = SessionLocal()
    db.query(ChatSession).filter(ChatSession.id == session_id).delete()
    db.commit()
    db.close()