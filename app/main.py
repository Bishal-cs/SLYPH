"""
S.Y.L.P.H MAIN API
============================


This module defines the FastAPI application and all HTTP endpoints. It is
designed for single-user use: one person runs one server (i.e., python run.py)
and uses it as their personal S.Y.L.P.H backend. Many people can each run
their own copy of this code on their own machine.

ENDPOINTS:
GET /                           - Returns API name and list of endpoints.
GET /health                     - Returns status of all services (for monitoring).
POST /chat                      - General chat pure LLM, no web search. Uses learning data.
POST /chat/realtime             - Realtime chat: runs a Tavily web search first, then
                                 sends results + context to Groq. Same session as /chat.
GET /chat/history/{id}          - Returns all messages for a session (general + realtime).

SESSION:
    Both /chat and /chat/realtime use the same session_id. If you omit session_id,
    the server generates a UUID and returns it; send it back on the next request
    to continue the conversation. Sessions are saved to disk and survive restarts.
STARTUP:
    On startup, the lifespan function builds the vector store from learning_data/*.txt
    and chats_data/*.json, then creates Groq, Realtime, and Chat services. On shutdown,
    it saves all in-memory sessions to disk.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
import asyncio
import webbrowser
import urllib.request
import urllib.error
from fastapi import Request
from app.models import ChatRequest, ChatResponse

#user-frendly message When groq rate limit (daily token quota) in exceeded.
RATE_LIMIT_MESSAGE = (
    "you've reached your daily API limit for this assistant."
    "your credits will reset in a few hours, or you can Upgrade Your Plan for more"
    "Please try agen later."
)

def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in str(exc) or "rate limit" in msg or "tokens per day" in msg

from app.services.vector_store import VectorStoreService
from app.services.groq_service import GroqService
from app.services.realtime_service import RealtimeGroqService
from app.services.chat_service import ChatServices
from app.config import VECTOR_STORE_DIR

from langchain_community.vectorstores import FAISS

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("S.Y.L.P.H")


def is_url_accessible(url: str = "http://localhost:8000/") -> bool:
    """Check if the local server URL is already reachable (to avoid duplicate browser open)."""
    try:
        with urllib.request.urlopen(url, timeout=1) as resp:
            return resp.status == 200
    except urllib.error.URLError:
        return False
    except Exception:
        return False

# -----------------------------------------------------------------------------
# GLOBAL SERVICE REFERENCES
# -----------------------------------------------------------------------------

vector_store_service: VectorStoreService = None
groq_service : GroqService = None
realtime_service: RealtimeGroqService = None
chat_service: ChatServices = None

def print_title():
    """Print the S.Y.L.P.H ASCII art title."""
    title = """

         _______   ___     ____  _   _ 
        / ___ \ \ / / |   |  _ \| | | |
        \___ \ \ V /| |   | |_) | |_| |
         ___) | | | | |___|  __/|  _  |
        |____/  |_| |_____|_|   |_| |_|
                        
                                                                                                                    
    """

    print(title)



@asynccontextmanager
async def lifespan(app: FastAPI):
    """"""
    global vector_store_service, groq_service, realtime_service, chat_service

    print_title()
    logger.info("="*60)
    logger.info("S.Y.L.P.H - Starting Up...........")
    logger.info("="*60)

    try:
        logger.info("Initializing vector store service.........")
        vector_store_service = VectorStoreService()
        vector_store_service.create_vector_store()
        logger.info("vector store Initialized Successfully")

        logger.info("Initializing Groq service (general Querires).......")
        groq_service = GroqService(vector_store_service)
        logger.info("Groq service Initialized successfully")

        logger.info("Initializing realtime groq service (with tavil search)......")
        realtime_service = RealtimeGroqService(vector_store_service)
        logger.info("Realtime Groq services Initialized Successfully")

        logger.info("Initializing chat Service...........")
        chat_service = ChatServices(groq_service, realtime_service)
        logger.info("Chat service Initialized Successfullly")

        logger.info("="*60)
        logger.info("Service status:")
        logger.info("   - Vector Store: Ready")
        logger.info("   - Groq AI (Gerneal): Ready")
        logger.info("   - Groq AI (Realtime): Ready")
        logger.info("   - Chat Service: Ready")
        logger.info("="*60)
        logger.info("S.Y.L.P.H is online and ready")
        logger.info("API http://localhost:8000")
        logger.info("="*60)

        # Auto-open browser when the site is not already reachable
        if not is_url_accessible("http://localhost:8000/"):
            webbrowser.open("http://localhost:8000")
            logger.info("Opened browser to http://localhost:8000")
        else:
            logger.info("Local URL http://localhost:8000 already accessible; skipping browser open.")

        yield

        logger.info("\nShutting down S.Y.L.P.H..............")
        if chat_service:
            for session_id in list(chat_service.sessions.keys()):
                chat_service.save_chat_session(session_id)
        logger.info("All sessions saved. Goodbye!")
    
    except Exception as e:
        logger.error(f"Fatal error during startup: {e}",exc_info=True)
        raise

app = FastAPI(
    title="S.Y.L.P.H API",
    description="Just A Rather very Intelligent System",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ✅ CORRECT
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "S.Y.L.P.H API",
        "endpoints": {
            "/chat": "General chat (pure LLM, no web search)",
            "/chat/realtime": "Realtime chat (with tavil search)",
            "/chat/history/{session_id}": "Get chat history",
            "/health": "System Health check"
        }
    }

def _health_payload():
    return {
        "status":"healthy",
        "vector_store": vector_store_service is not None,
        "groq_service": groq_service is not None,
        "realtime_service": realtime_service is not None,
        "chat_service": chat_service is not None
    }

@app.get("/health")
async def health():
    return _health_payload()

@app.get("/api/health")
async def api_health():
    return _health_payload()

@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(request: ChatRequest):
    return await chat(request)

@app.post("/api/chat/realtime", response_model=ChatResponse)
async def api_chat_realtime(request: ChatRequest):
    return await chat_realtime(request)

@app.post("/api/chat/stream")
async def api_chat_stream(request: ChatRequest):
    return await chat_stream(request)

@app.post("/api/chat/realtime/stream")
async def api_chat_realtime_stream(request: ChatRequest):
    return await chat_realtime_stream(request)

@app.post("/api/chat/Sylph/stream")
async def api_chat_sylph_stream(request: ChatRequest):
    return await chat_sylph_stream(request)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat Service Not Initialized")

    try:
        session_id = chat_service.get_or_create_session(request.session_id)
        response_text = chat_service.process_message(session_id, request.message)
        chat_service.save_chat_session(session_id)

        return ChatResponse(response=response_text, session_id=session_id)

    except ValueError as e:
        logger.warning(f"Invalid session_id: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        if _is_rate_limit_error(e):
            logger.warning(f"Rate limit hit: {e}")
            raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)

        logger.error(f"Error Processing Chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

    
@app.post("/chat/realtime", response_model=ChatResponse)
async def chat_realtime(request: ChatRequest):
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat Service Not Initialized")
    
    if not realtime_service:
        raise HTTPException(status_code=503, detail="Realtime Service Not Initialized")
    
    try:
        session_id = chat_service.get_or_create_session(request.session_id)
        response_text = chat_service.process_realtime_message(session_id, request.message)
        chat_service.save_chat_session(session_id)
        return ChatResponse(response=response_text, session_id=session_id)
    
    except ValueError as e:
        logger.warning(f"Invalid session_id: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        if _is_rate_limit_error(e):
            logger.warning(f"Rate limit hit: {e}")
            raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)
        logger.error(f"Error processing realtime chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")
    
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat Service Not Initialized")

    async def generate():
        try:
            session_id = chat_service.get_or_create_session(request.session_id)
            response_text = await asyncio.get_event_loop().run_in_executor(None, chat_service.process_message, session_id, request.message)
            await asyncio.get_event_loop().run_in_executor(None, chat_service.save_chat_session, session_id)
            # Yield response as Server-Sent Events
            for word in response_text.split():
                yield f"data: {word}\n\n"
                await asyncio.sleep(0.05)  # Small delay for streaming effect
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/chat/realtime/stream")
async def chat_realtime_stream(request: ChatRequest):
    if not chat_service:
        raise HTTPException(status_code=503, detail="chat service not initialized")
    
    if not realtime_service:
        raise HTTPException(status_code=503, detail="Realtime Service Not initialized")

    async def generate():
        try:
            session_id = chat_service.get_or_create_session(request.session_id)
            response_text = await asyncio.get_event_loop().run_in_executor(None, chat_service.process_realtime_message, session_id, request.message)
            await asyncio.get_event_loop().run_in_executor(None, chat_service.save_chat_session, session_id)
            for word in response_text.split():
                yield f"data: {word}\n\n"
                await asyncio.sleep(0.05)
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/chat/Sylph/stream")
async def chat_sylph_stream(request: ChatRequest):
    # Auto-route based on message content or something, but for now, use general
    return await chat_stream(request)
    
@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat Service Not Initialized")
    
    try:
        messages = chat_service.get_chat_history(session_id)
        return {
            "session_id": session_id,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")
    

def run():
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    run()