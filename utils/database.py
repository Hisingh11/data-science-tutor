from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    chats = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")

class ChatSession(Base):
    __tablename__ = 'chat_sessions'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    title = Column(String(200), default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = relationship("User", back_populates="chats")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('chat_sessions.id'))
    role = Column(String(20))
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    attachments = Column(Text)
    session = relationship("ChatSession", back_populates="messages")

os.makedirs("./data", exist_ok=True)
DATABASE_URL = "sqlite:///./data/ds_tutor.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(engine)