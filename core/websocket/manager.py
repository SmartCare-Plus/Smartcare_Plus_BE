"""
SMARTCARE+ WebSocket Connection Manager

Manages WebSocket connections with authentication,
connection pooling, and broadcasting capabilities.
"""

import asyncio
import logging
import json
from typing import Dict, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from core.config import settings

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """WebSocket message types."""
    # System messages
    PING = "ping"
    PONG = "pong"
    AUTH = "auth"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILED = "auth_failed"
    ERROR = "error"
    
    # Video streaming
    VIDEO_FRAME = "video_frame"
    FRAME_PROCESSED = "frame_processed"
    
    # Physio events
    GAIT_UPDATE = "gait_update"
    POSE_UPDATE = "pose_update"
    EXERCISE_UPDATE = "exercise_update"
    
    # Guardian events
    FALL_DETECTED = "fall_detected"
    SOS_ALERT = "sos_alert"
    GEOFENCE_BREACH = "geofence_breach"
    STATUS_UPDATE = "status_update"
    
    # General
    NOTIFICATION = "notification"
    DATA = "data"


@dataclass
class WebSocketMessage:
    """Structured WebSocket message."""
    type: MessageType
    payload: Any = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_json(self) -> str:
        return json.dumps({
            "type": self.type.value if isinstance(self.type, MessageType) else self.type,
            "payload": self.payload,
            "timestamp": self.timestamp
        })
    
    @classmethod
    def from_json(cls, data: str) -> "WebSocketMessage":
        parsed = json.loads(data)
        return cls(
            type=parsed.get("type", "data"),
            payload=parsed.get("payload"),
            timestamp=parsed.get("timestamp", datetime.now(timezone.utc).isoformat())
        )


@dataclass
class ConnectedClient:
    """Represents a connected WebSocket client."""
    websocket: WebSocket
    user_id: str
    client_id: str
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    subscriptions: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_connected(self) -> bool:
        return self.websocket.client_state == WebSocketState.CONNECTED


class ConnectionManager:
    """
    Manages all WebSocket connections.
    
    Features:
    - Connection pooling by user/client ID
    - Room-based subscriptions (e.g., "elderly:123", "physio:session:456")
    - Broadcasting to specific users or rooms
    - Heartbeat/ping monitoring
    """
    
    def __init__(self, max_connections: int = None):
        self.max_connections = max_connections or settings.WS_MAX_CONNECTIONS
        
        # Active connections: client_id -> ConnectedClient
        self._connections: Dict[str, ConnectedClient] = {}
        
        # User index: user_id -> set of client_ids
        self._user_connections: Dict[str, Set[str]] = {}
        
        # Room subscriptions: room_id -> set of client_ids
        self._rooms: Dict[str, Set[str]] = {}
        
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
        
        # Heartbeat task
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        logger.info(f"🔌 ConnectionManager initialized (max: {self.max_connections})")
    
    @property
    def connection_count(self) -> int:
        return len(self._connections)
    
    async def connect(
        self,
        websocket: WebSocket,
        user_id: str,
        client_id: str = None
    ) -> ConnectedClient:
        """
        Accept a new WebSocket connection.
        """
        # Check connection limit
        if self.connection_count >= self.max_connections:
            await websocket.close(code=1013, reason="Server at capacity")
            raise ConnectionError("Maximum connections reached")
        
        await websocket.accept()
        
        client_id = client_id or f"{user_id}_{datetime.now().timestamp()}"
        
        client = ConnectedClient(
            websocket=websocket,
            user_id=user_id,
            client_id=client_id
        )
        
        async with self._lock:
            # Store connection
            self._connections[client_id] = client
            
            # Index by user
            if user_id not in self._user_connections:
                self._user_connections[user_id] = set()
            self._user_connections[user_id].add(client_id)
        
        logger.info(f"✅ Client connected: {client_id} (user: {user_id})")
        
        # Send connection confirmation
        await self.send_to_client(client_id, WebSocketMessage(
            type=MessageType.AUTH_SUCCESS,
            payload={"client_id": client_id, "user_id": user_id}
        ))
        
        return client
    
    async def disconnect(self, client_id: str):
        """Disconnect and cleanup a client."""
        async with self._lock:
            client = self._connections.pop(client_id, None)
            
            if client:
                # Remove from user index
                if client.user_id in self._user_connections:
                    self._user_connections[client.user_id].discard(client_id)
                    if not self._user_connections[client.user_id]:
                        del self._user_connections[client.user_id]
                
                # Remove from all rooms
                for room_id in list(client.subscriptions):
                    if room_id in self._rooms:
                        self._rooms[room_id].discard(client_id)
                        if not self._rooms[room_id]:
                            del self._rooms[room_id]
                
                logger.info(f"👋 Client disconnected: {client_id}")
    
    async def subscribe(self, client_id: str, room_id: str):
        """Subscribe a client to a room."""
        async with self._lock:
            client = self._connections.get(client_id)
            if not client:
                return
            
            client.subscriptions.add(room_id)
            
            if room_id not in self._rooms:
                self._rooms[room_id] = set()
            self._rooms[room_id].add(client_id)
            
            logger.debug(f"Client {client_id} subscribed to {room_id}")
    
    async def unsubscribe(self, client_id: str, room_id: str):
        """Unsubscribe a client from a room."""
        async with self._lock:
            client = self._connections.get(client_id)
            if client:
                client.subscriptions.discard(room_id)
            
            if room_id in self._rooms:
                self._rooms[room_id].discard(client_id)
                if not self._rooms[room_id]:
                    del self._rooms[room_id]
    
    async def send_to_client(self, client_id: str, message: WebSocketMessage) -> bool:
        """Send a message to a specific client."""
        client = self._connections.get(client_id)
        
        if not client or not client.is_connected():
            return False
        
        try:
            await client.websocket.send_text(message.to_json())
            client.last_activity = datetime.now(timezone.utc)
            return True
        except Exception as e:
            logger.error(f"Failed to send to {client_id}: {e}")
            await self.disconnect(client_id)
            return False
    
    async def send_to_user(self, user_id: str, message: WebSocketMessage) -> int:
        """Send a message to all connections of a user."""
        client_ids = self._user_connections.get(user_id, set())
        sent_count = 0
        
        for client_id in list(client_ids):
            if await self.send_to_client(client_id, message):
                sent_count += 1
        
        return sent_count
    
    async def broadcast_to_room(self, room_id: str, message: WebSocketMessage) -> int:
        """Broadcast a message to all clients in a room."""
        client_ids = self._rooms.get(room_id, set())
        sent_count = 0
        
        for client_id in list(client_ids):
            if await self.send_to_client(client_id, message):
                sent_count += 1
        
        return sent_count
    
    async def broadcast_all(self, message: WebSocketMessage) -> int:
        """Broadcast a message to all connected clients."""
        sent_count = 0
        
        for client_id in list(self._connections.keys()):
            if await self.send_to_client(client_id, message):
                sent_count += 1
        
        return sent_count
    
    async def handle_message(
        self,
        client_id: str,
        raw_message: str,
        handler: Callable[[str, WebSocketMessage], Any] = None
    ):
        """Process an incoming message from a client."""
        try:
            message = WebSocketMessage.from_json(raw_message)
            
            # Update last activity
            client = self._connections.get(client_id)
            if client:
                client.last_activity = datetime.now(timezone.utc)
            
            # Handle ping
            if message.type == MessageType.PING.value:
                await self.send_to_client(client_id, WebSocketMessage(type=MessageType.PONG))
                return
            
            # Custom handler
            if handler:
                await handler(client_id, message)
                
        except json.JSONDecodeError:
            await self.send_to_client(client_id, WebSocketMessage(
                type=MessageType.ERROR,
                payload={"error": "Invalid JSON"}
            ))
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")
    
    async def start_heartbeat(self, interval: int = None):
        """Start heartbeat task to check connection health."""
        interval = interval or settings.WS_HEARTBEAT_INTERVAL
        
        async def heartbeat_loop():
            while True:
                await asyncio.sleep(interval)
                await self._check_connections()
        
        self._heartbeat_task = asyncio.create_task(heartbeat_loop())
        logger.info(f"💓 Heartbeat started (interval: {interval}s)")
    
    async def stop_heartbeat(self):
        """Stop the heartbeat task."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
    
    async def _check_connections(self):
        """Check all connections and disconnect dead ones."""
        for client_id in list(self._connections.keys()):
            client = self._connections.get(client_id)
            if client and not client.is_connected():
                await self.disconnect(client_id)
    
    def get_stats(self) -> dict:
        """Get connection statistics."""
        return {
            "total_connections": self.connection_count,
            "unique_users": len(self._user_connections),
            "rooms": len(self._rooms),
            "max_connections": self.max_connections
        }


# Global connection manager instance
connection_manager = ConnectionManager()


async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
    handler: Callable[[str, WebSocketMessage], Any] = None
):
    """
    Reusable WebSocket endpoint handler.
    
    Usage in router:
        @router.websocket("/ws/{user_id}")
        async def ws_route(websocket: WebSocket, user_id: str):
            await websocket_endpoint(websocket, user_id, my_handler)
    """
    client = await connection_manager.connect(websocket, user_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            await connection_manager.handle_message(client.client_id, data, handler)
    
    except WebSocketDisconnect:
        await connection_manager.disconnect(client.client_id)
    
    except Exception as e:
        logger.error(f"WebSocket error for {client.client_id}: {e}")
        await connection_manager.disconnect(client.client_id)
