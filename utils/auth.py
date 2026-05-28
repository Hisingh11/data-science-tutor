"""
User Authentication and Profile Management
"""

import streamlit as st
import hashlib
import json
import os
from datetime import datetime
from typing import Dict, Optional

class UserAuth:
    def __init__(self, users_file="./data/users.json"):
        self.users_file = users_file
        os.makedirs(os.path.dirname(users_file), exist_ok=True)
        self._init_users_file()
    
    def _init_users_file(self):
        if not os.path.exists(self.users_file):
            initial_users = {
                "demo": {
                    "password": hashlib.sha256("demo123".encode()).hexdigest(),
                    "created_at": datetime.now().isoformat(),
                    "name": "Demo User",
                    "email": "demo@example.com"
                }
            }
            with open(self.users_file, 'w') as f:
                json.dump(initial_users, f, indent=2)
    
    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register(self, username: str, password: str, name: str, email: str) -> Dict:
        if not os.path.exists(self.users_file):
            users = {}
        else:
            with open(self.users_file, 'r') as f:
                users = json.load(f)
        
        if username in users:
            return {"success": False, "error": "Username already exists"}
        
        users[username] = {
            "password": self._hash_password(password),
            "created_at": datetime.now().isoformat(),
            "name": name,
            "email": email,
            "last_login": None
        }
        
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=2)
        
        return {"success": True, "message": "Registration successful"}
    
    def login(self, username: str, password: str) -> Dict:
        with open(self.users_file, 'r') as f:
            users = json.load(f)
        
        if username not in users:
            return {"success": False, "error": "User not found"}
        
        if users[username]["password"] != self._hash_password(password):
            return {"success": False, "error": "Invalid password"}
        
        # Update last login
        users[username]["last_login"] = datetime.now().isoformat()
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=2)
        
        return {
            "success": True, 
            "user": {
                "username": username,
                "name": users[username]["name"],
                "email": users[username]["email"]
            }
        }
    
    def logout(self):
        for key in ["user", "chat_history", "current_session"]:
            if key in st.session_state:
                del st.session_state[key]

class ChatHistoryManager:
    def __init__(self, history_dir="./data/chat_history"):
        self.history_dir = history_dir
        os.makedirs(history_dir, exist_ok=True)
    
    def save_conversation(self, username: str, conversation: list):
        file_path = os.path.join(self.history_dir, f"{username}.json")
        
        existing = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                existing = json.load(f)
        
        existing.append({
            "timestamp": datetime.now().isoformat(),
            "conversation": conversation
        })
        
        # Keep only last 50 conversations
        if len(existing) > 50:
            existing = existing[-50:]
        
        with open(file_path, 'w') as f:
            json.dump(existing, f, indent=2)
    
    def load_conversations(self, username: str) -> list:
        file_path = os.path.join(self.history_dir, f"{username}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return []
    
    def delete_conversation(self, username: str, index: int):
        file_path = os.path.join(self.history_dir, f"{username}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                conversations = json.load(f)
            
            if 0 <= index < len(conversations):
                conversations.pop(index)
            
            with open(file_path, 'w') as f:
                json.dump(conversations, f, indent=2)