import logging
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Annotated
from pathlib import Path
import secrets

from fastapi import (
    FastAPI, APIRouter, Depends, HTTPException, status,
    WebSocket, WebSocketDisconnect
)
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import auth, game_logic, state_manager
from .websocket_manager import manager as websocket_manager
from .config import settings

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- App Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Application startup...")
    await state_manager.init_storage()
    state_manager.start_auto_save_task()
    yield
    logging.info("Application shutdown...")
    await state_manager.shutdown_storage()

# --- FastAPI App Instance ---
app = FastAPI(lifespan=lifespan, title="浮生十梦")

# --- Routers ---
# Router for /api prefixed routes
api_router = APIRouter(prefix="/api")


# --- Authentication Routes ---
class PasswordLoginRequest(BaseModel):
    password: str


@api_router.post("/login")
async def login(payload: PasswordLoginRequest):
    if not settings.GAME_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server is not configured for password login",
        )

    if not secrets.compare_digest(payload.password, settings.GAME_PASSWORD):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password",
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    jwt_payload = {
        "sub": settings.PLAYER_ID,
        "id": 1,
        "name": settings.PLAYER_ID,
        "trust_level": 0,
    }
    access_token = auth.create_access_token(
        data=jwt_payload, expires_delta=access_token_expires
    )

    response = JSONResponse({"ok": True})
    response.set_cookie(
        "token",
        value=access_token,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        max_age=int(access_token_expires.total_seconds()),
        samesite="lax",
    )
    return response


@api_router.post("/logout")
async def logout():
    """
    Logs the user out by clearing the authentication cookie.
    """
    response = RedirectResponse(url="/")
    response.delete_cookie("token")
    return response

# --- Game Routes ---
@api_router.post("/game/init")
async def init_game(
    current_user: Annotated[dict, Depends(auth.get_current_active_user)],
):
    """
    Initializes or retrieves the daily game session for the player.
    This does NOT start a trial, it just ensures the session for the day exists.
    """
    game_state = await game_logic.get_or_create_daily_session(current_user)
    return game_state

# --- WebSocket Endpoint ---
@api_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles WebSocket connections for real-time game state updates."""
    token = websocket.cookies.get("token")
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Missing token")
        return
    try:
        payload = auth.decode_access_token(token)
        username: str | None = payload.get("sub")
        if username is None:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token payload")
            return
    except HTTPException:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Token validation failed")
        return

    await websocket_manager.connect(websocket, username)

    try:
        user_info = await auth.get_current_user(token)
        session = await state_manager.get_session(user_info["username"])
        if session:
            await websocket_manager.send_json_to_player(
                user_info["username"], {"type": "full_state", "data": session}
            )

        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            if action:
                await game_logic.process_player_action(user_info, action)

    except WebSocketDisconnect:
        websocket_manager.disconnect(username)


# --- Include API Router and Mount Static Files ---
app.include_router(api_router)
static_files_dir = Path(__file__).parent.parent.parent / "frontend"
app.mount("/", StaticFiles(directory=static_files_dir, html=True), name="static")

# --- Uvicorn Runner ---
if __name__ == "__main__":
    import uvicorn
    # The first argument should be "main:app" and we should specify the app_dir
    # This makes running the script directly more robust.
    # For command line, the equivalent is:
    # uvicorn backend.app.main:app --host <host> --port <port> --reload
    uvicorn.run(
        "main:app",
        app_dir="backend/app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.UVICORN_RELOAD
    )