"""
WebSocket handlers for real-time ECG streaming
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import redis.asyncio as redis
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from app.core.config import settings
from app.tasks.ecg_tasks import process_streaming_ecg_sample, cleanup_streaming_session

logger = logging.getLogger(__name__)


class ECGStreamingManager:
    """Manages WebSocket connections for real-time ECG streaming"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
        self.redis_client: Optional[redis.Redis] = None
        
    async def initialize_redis(self) -> None:
        """Initialize Redis connection for pub/sub"""
        try:
            self.redis_client = redis.from_url(settings.REDIS_URL)
            await self.redis_client.ping()
            logger.info("Redis connection established for ECG streaming")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            
    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """Accept WebSocket connection and start streaming session"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        
        self.session_metadata[session_id] = {
            "connected_at": asyncio.get_event_loop().time(),
            "samples_processed": 0,
            "last_heartbeat": asyncio.get_event_loop().time()
        }
        
        asyncio.create_task(self._subscribe_to_session(session_id))
        
        logger.info(f"ECG streaming session connected: {session_id}")
        
    async def disconnect(self, session_id: str) -> None:
        """Disconnect WebSocket and cleanup session"""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
            del self.active_connections[session_id]
            
        if session_id in self.session_metadata:
            del self.session_metadata[session_id]
            
        cleanup_streaming_session.delay(session_id)
        
        logger.info(f"ECG streaming session disconnected: {session_id}")
        
    async def send_to_session(self, session_id: str, message: Dict[str, Any]) -> bool:
        """Send message to specific session"""
        if session_id not in self.active_connections:
            return False
            
        websocket = self.active_connections[session_id]
        
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps(message))
                return True
            else:
                await self.disconnect(session_id)
                return False
        except Exception as e:
            logger.error(f"Failed to send message to session {session_id}: {e}")
            await self.disconnect(session_id)
            return False
            
    async def broadcast_to_all(self, message: Dict[str, Any]) -> int:
        """Broadcast message to all active sessions"""
        sent_count = 0
        disconnected_sessions = []
        
        for session_id, websocket in self.active_connections.items():
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps(message))
                    sent_count += 1
                else:
                    disconnected_sessions.append(session_id)
            except Exception as e:
                logger.error(f"Failed to broadcast to session {session_id}: {e}")
                disconnected_sessions.append(session_id)
                
        for session_id in disconnected_sessions:
            await self.disconnect(session_id)
            
        return sent_count
        
    async def process_ecg_data(
        self, 
        session_id: str, 
        ecg_data: List[float],
        timestamp: float,
        lead_name: str = "II"
    ) -> None:
        """Process incoming ECG data and trigger analysis"""
        try:
            if session_id in self.session_metadata:
                self.session_metadata[session_id]["samples_processed"] += len(ecg_data)
                self.session_metadata[session_id]["last_heartbeat"] = asyncio.get_event_loop().time()
                
            process_streaming_ecg_sample.delay(
                session_id, ecg_data, timestamp, lead_name
            )
            
        except Exception as e:
            logger.error(f"Failed to process ECG data for session {session_id}: {e}")
            await self.send_to_session(session_id, {
                "type": "error",
                "message": f"Processing failed: {str(e)}",
                "timestamp": timestamp
            })
            
    async def _subscribe_to_session(self, session_id: str) -> None:
        """Subscribe to Redis channel for session results"""
        if not self.redis_client:
            await self.initialize_redis()
            
        if not self.redis_client:
            logger.error("Redis not available for session subscription")
            return
            
        try:
            pubsub = self.redis_client.pubsub()
            channel = f"ecg_stream:{session_id}"
            await pubsub.subscribe(channel)
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        await self.send_to_session(session_id, {
                            "type": "analysis_result",
                            "data": data
                        })
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode Redis message: {e}")
                    except Exception as e:
                        logger.error(f"Failed to process Redis message: {e}")
                        
                if session_id not in self.active_connections:
                    break
                    
        except Exception as e:
            logger.error(f"Redis subscription failed for session {session_id}: {e}")
        finally:
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.close()
            except Exception as e:
                logger.error(f"Failed to cleanup Redis subscription: {e}")
                
    async def send_heartbeat(self, session_id: str) -> bool:
        """Send heartbeat to maintain connection"""
        return await self.send_to_session(session_id, {
            "type": "heartbeat",
            "timestamp": asyncio.get_event_loop().time(),
            "session_id": session_id
        })
        
    async def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a streaming session"""
        if session_id not in self.session_metadata:
            return None
            
        metadata = self.session_metadata[session_id]
        current_time = asyncio.get_event_loop().time()
        
        return {
            "session_id": session_id,
            "connected": session_id in self.active_connections,
            "duration_seconds": current_time - metadata["connected_at"],
            "samples_processed": metadata["samples_processed"],
            "last_heartbeat": metadata["last_heartbeat"],
            "time_since_last_heartbeat": current_time - metadata["last_heartbeat"]
        }
        
    async def get_all_sessions_stats(self) -> Dict[str, Any]:
        """Get statistics for all active sessions"""
        stats = {
            "total_sessions": len(self.active_connections),
            "sessions": {}
        }
        
        for session_id in self.active_connections.keys():
            session_stats = await self.get_session_stats(session_id)
            if session_stats:
                stats["sessions"][session_id] = session_stats
                
        return stats
        
    async def cleanup_inactive_sessions(self, timeout_seconds: int = 300) -> int:
        """Clean up sessions that haven't sent heartbeat recently"""
        current_time = asyncio.get_event_loop().time()
        inactive_sessions = []
        
        for session_id, metadata in self.session_metadata.items():
            if current_time - metadata["last_heartbeat"] > timeout_seconds:
                inactive_sessions.append(session_id)
                
        for session_id in inactive_sessions:
            await self.disconnect(session_id)
            
        return len(inactive_sessions)


streaming_manager = ECGStreamingManager()


async def handle_ecg_websocket(websocket: WebSocket, session_id: str) -> None:
    """Handle ECG streaming WebSocket connection"""
    await streaming_manager.connect(websocket, session_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            
            if message_type == "ecg_data":
                await streaming_manager.process_ecg_data(
                    session_id,
                    message["data"],
                    message["timestamp"],
                    message.get("lead", "II")
                )
                
            elif message_type == "heartbeat":
                await streaming_manager.send_heartbeat(session_id)
                
            elif message_type == "get_stats":
                stats = await streaming_manager.get_session_stats(session_id)
                await streaming_manager.send_to_session(session_id, {
                    "type": "stats",
                    "data": stats
                })
                
            else:
                await streaming_manager.send_to_session(session_id, {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        await streaming_manager.disconnect(session_id)
