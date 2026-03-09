"""
CHAT SERVICE MODULE
====================

This service owns all chat sessions and conversations logic. It is used by the /chat and /chat/realtime endpoints. Design for single-user use: one server has one Chatservice and one in-memory section store; the user can have many sections (each identified by a section_id).

RESPONSIBILITIES:
 - get_or_create_session(session_id): Return existing session or create new one.
 - If the user sends a session_id that was used before (e.g. before a restart),
 - we try to load it from disk so the conversation continues.
 - add_message / get_chat_history: Keep messages in memory per session.
 - format_history_for_1lm: Turn the message list into (user, assistant) pairs
 - and trim to MAX_CHAT_HISTORY_TURNS so we don't overflow the prompt.
 - process_message / process_realtime_message: Add user message, call Groq (or
 - RealtimeGroq), add assistant reply, return reply.
 - save_chat_session: Write session to database/chats_data/ *. json so it persists -
 - and can be loaded on next startup (and used by the vector store for retrieval).

"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict
import uuid

from config import CHATS_DATA_DIR, MAX_CHAT_HISTORY_TURNS
from app.models import ChatMessage, ChatHistory
from app.services.groq_service import GroqService
from app.services.realtime_service import RealtimeGroqService

logger = logging.getLogger("Z.U.R.I")

# ========================================
# CHAT SERVICE
# ========================================

class ChatService:
    """
    Manages chat sessions: in-memory message lists, load/save to disk, and calling Groq (or Realtime) to get replies. All state for active sessions is in self.sessions; saving to disk is done after each message so conversations survive restarts.
    """

    def __init__(self, groq_service: GroqService, realtime_service: RealtimeGroqService = None):
        """Store references to the Groq and Realtime services; keep sessions in memory."""
        self.groq_service = groq_service
        self.realtime_service = realtime_service
        # Map: session_id -> list of ChatMessage (user and assistant messages in order).
        self.sessions: Dict[str, List[ChatMessage]] = {}

    # ----------------------------------------------------
    # SESSION LOAD / VALIDATE / GET-OR-CREATE
    # ----------------------------------------------------

    def load_session_from_disk(self, session_id: str) -> bool:
        """
        Load a session from database/chats data/ if a file for this session_id exists.

        File name is chat_{safe_session_id}.json where safe_session_id has dashes/spaces removed.
        On success we put the messages into self.sessions[session_id] so later requests use them.
        Returns True it loaded, False if file missing or unreadable.
        """
        # Sanitize ID for use in filename (no dashes or spaces).
        safe_session_id = session_id.replace("-", "").replace(" ", "_")
        filename = f"chat_{safe_session_id}.json"
        filepath = CHATS_DATA_DIR / filename

        if not filepath.exists():
            return False
        
        try:
            with open(filepath, "r") as f:
                chat_dict = json.load(f)
            # Convert dict to list of ChatMessage objects.
            messages = [
                ChatMessage(role=msg.get("role"), content = msg.get("content"))
                for msg in chat_dict.get("messages", [])
            ]
            self.sessions[session_id] = messages
            return True
        except Exception as e:
            logger.error("Failed to load session %s from disk: %s", session_id, e)
            return False
        
    def validate_session_id(self, session_id: str) -> bool:
        """
        Return True if session_id is safe to use (non-empty, no path traversal, length <= 255). Used to reject molicious or invalid IDs before we use them in file paths.
        """
        if not session_id or not session_id.strip():
            return False
        # Block path traversal and path separators.
        if ".." in session_id or "/" in session_id or "\\" in session_id:
            return False
        if len(session_id) > 255:
            return False
        return True
    
    def get_or_create_session(self, session_id: Optional[str]) -> str:
        """
        Return a session ID and ensure that session exists in memory.

        - If session_id is None: create a new session with a new UUID and return it.
        - If session_id is provided: validate it; if it's in self.sessions return it; 
          else try to load from disk; if not found. create new session with that ID.
        Raises ValueError if session_id is invalid (empty, path traversal, too long)
        """
        if not session_id:
            new_session_id = str(uuid.uuid4())
            self.sessions[new_session_id] = []
            return new_session_id
        
        if not self.validate_session_id(session_id):
            raise ValueError(
                f"Invalid session_id fromat {session_id}. Session ID must be non-empty, "
                "not contain path traversal characters, and be under 255 characters."
            )
        
        if session_id in self.sessions:
            return session_id

        if self.load_session_from_disk(session_id):
            return session_id

        # New session with this id (e.g. client sent and ID that was never saved).
        self.sessions[session_id] = []
        return session_id
    
    # ----------------------------------------------------
    # MESSAGE AND HISTORY FORMATTING
    # ----------------------------------------------------
    
        

            