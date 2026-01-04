"""
SMARTCARE+ Firebase Cloud Messaging (FCM) Service

Push notification sender for alerts, reminders, and updates.
"""

import logging
from typing import Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timezone

from firebase_admin import messaging
from core.database import init_firebase, is_mock_mode

logger = logging.getLogger(__name__)


class NotificationType(str, Enum):
    """Types of push notifications."""
    # Guardian alerts (high priority)
    FALL_DETECTED = "fall_detected"
    SOS_ALERT = "sos_alert"
    GEOFENCE_EXIT = "geofence_exit"
    INACTIVITY_ALERT = "inactivity_alert"
    
    # Health updates (normal priority)
    MEDICATION_REMINDER = "medication_reminder"
    HYDRATION_REMINDER = "hydration_reminder"
    EXERCISE_REMINDER = "exercise_reminder"
    
    # General (low priority)
    DAILY_SUMMARY = "daily_summary"
    SYSTEM_UPDATE = "system_update"


@dataclass
class PushNotification:
    """Push notification payload."""
    title: str
    body: str
    notification_type: NotificationType
    data: Optional[Dict[str, str]] = None
    image_url: Optional[str] = None
    click_action: Optional[str] = None
    
    def is_high_priority(self) -> bool:
        return self.notification_type in [
            NotificationType.FALL_DETECTED,
            NotificationType.SOS_ALERT,
            NotificationType.GEOFENCE_EXIT
        ]


class FCMService:
    """
    Firebase Cloud Messaging service for push notifications.
    
    Features:
    - Send to individual devices
    - Send to topics (e.g., all guardians of an elderly)
    - Send to multiple devices
    - High/normal priority handling
    """
    
    def __init__(self):
        self._initialized = False
        self._mock_mode = False
        self._sent_count = 0
        self._failed_count = 0
        
    def initialize(self):
        """Initialize FCM (requires Firebase Admin SDK)."""
        if self._initialized:
            return
        
        init_firebase()
        self._mock_mode = is_mock_mode()
        self._initialized = True
        
        if self._mock_mode:
            logger.warning("🧪 FCM running in MOCK MODE - notifications will be logged only")
        else:
            logger.info("🔔 FCM Service initialized")
    
    def _build_message(
        self,
        notification: PushNotification,
        token: str = None,
        topic: str = None
    ) -> messaging.Message:
        """Build FCM message object."""
        
        # Notification payload (shown in system tray)
        fcm_notification = messaging.Notification(
            title=notification.title,
            body=notification.body,
            image=notification.image_url
        )
        
        # Data payload (for app handling)
        data = notification.data or {}
        data["notification_type"] = notification.notification_type.value
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Android-specific config
        android_config = messaging.AndroidConfig(
            priority="high" if notification.is_high_priority() else "normal",
            notification=messaging.AndroidNotification(
                icon="ic_notification",
                color="#00F5FF",  # Neon cyan
                click_action=notification.click_action,
                channel_id="smartcare_alerts" if notification.is_high_priority() else "smartcare_general"
            )
        )
        
        # iOS-specific config
        apns_config = messaging.APNSConfig(
            payload=messaging.APNSPayload(
                aps=messaging.Aps(
                    alert=messaging.ApsAlert(
                        title=notification.title,
                        body=notification.body
                    ),
                    sound="default" if notification.is_high_priority() else None,
                    badge=1
                )
            )
        )
        
        # Build message
        message_kwargs = {
            "notification": fcm_notification,
            "data": data,
            "android": android_config,
            "apns": apns_config
        }
        
        if token:
            message_kwargs["token"] = token
        elif topic:
            message_kwargs["topic"] = topic
        
        return messaging.Message(**message_kwargs)
    
    async def send_to_device(
        self,
        device_token: str,
        notification: PushNotification
    ) -> Optional[str]:
        """
        Send notification to a specific device.
        
        Returns:
            Message ID on success, None on failure
        """
        self.initialize()
        
        if self._mock_mode:
            logger.info(
                f"📱 [MOCK] Notification to device: {device_token[:20]}...\n"
                f"   Title: {notification.title}\n"
                f"   Body: {notification.body}"
            )
            self._sent_count += 1
            return "mock_message_id"
        
        try:
            message = self._build_message(notification, token=device_token)
            response = messaging.send(message)
            
            self._sent_count += 1
            logger.info(f"✅ Notification sent: {response}")
            return response
            
        except messaging.UnregisteredError:
            logger.warning(f"Device token expired: {device_token[:20]}...")
            self._failed_count += 1
            return None
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            self._failed_count += 1
            return None
    
    async def send_to_topic(
        self,
        topic: str,
        notification: PushNotification
    ) -> Optional[str]:
        """
        Send notification to all subscribers of a topic.
        
        Topics: "elderly_<id>", "guardian_<id>", "all_users"
        """
        self.initialize()
        
        if self._mock_mode:
            logger.info(
                f"📢 [MOCK] Notification to topic: {topic}\n"
                f"   Title: {notification.title}\n"
                f"   Body: {notification.body}"
            )
            self._sent_count += 1
            return "mock_topic_message_id"
        
        try:
            message = self._build_message(notification, topic=topic)
            response = messaging.send(message)
            
            self._sent_count += 1
            logger.info(f"✅ Topic notification sent to '{topic}': {response}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to send topic notification: {e}")
            self._failed_count += 1
            return None
    
    async def send_to_multiple(
        self,
        device_tokens: List[str],
        notification: PushNotification
    ) -> Dict[str, Any]:
        """
        Send notification to multiple devices.
        
        Returns:
            Dict with success_count, failure_count, and failed_tokens
        """
        self.initialize()
        
        if self._mock_mode:
            logger.info(
                f"📱 [MOCK] Notification to {len(device_tokens)} devices\n"
                f"   Title: {notification.title}"
            )
            self._sent_count += len(device_tokens)
            return {
                "success_count": len(device_tokens),
                "failure_count": 0,
                "failed_tokens": []
            }
        
        if not device_tokens:
            return {"success_count": 0, "failure_count": 0, "failed_tokens": []}
        
        try:
            messages = [
                self._build_message(notification, token=token)
                for token in device_tokens
            ]
            
            response = messaging.send_each(messages)
            
            failed_tokens = []
            for idx, result in enumerate(response.responses):
                if not result.success:
                    failed_tokens.append(device_tokens[idx])
            
            self._sent_count += response.success_count
            self._failed_count += response.failure_count
            
            logger.info(
                f"✅ Batch notification: {response.success_count} sent, "
                f"{response.failure_count} failed"
            )
            
            return {
                "success_count": response.success_count,
                "failure_count": response.failure_count,
                "failed_tokens": failed_tokens
            }
            
        except Exception as e:
            logger.error(f"Batch notification failed: {e}")
            self._failed_count += len(device_tokens)
            return {
                "success_count": 0,
                "failure_count": len(device_tokens),
                "failed_tokens": device_tokens
            }
    
    async def subscribe_to_topic(
        self,
        device_tokens: List[str],
        topic: str
    ) -> bool:
        """Subscribe devices to a topic."""
        self.initialize()
        
        if self._mock_mode:
            logger.info(f"[MOCK] Subscribed {len(device_tokens)} devices to topic: {topic}")
            return True
        
        try:
            response = messaging.subscribe_to_topic(device_tokens, topic)
            logger.info(f"Subscribed to '{topic}': {response.success_count} success")
            return response.failure_count == 0
        except Exception as e:
            logger.error(f"Topic subscription failed: {e}")
            return False
    
    async def unsubscribe_from_topic(
        self,
        device_tokens: List[str],
        topic: str
    ) -> bool:
        """Unsubscribe devices from a topic."""
        self.initialize()
        
        if self._mock_mode:
            logger.info(f"[MOCK] Unsubscribed {len(device_tokens)} devices from topic: {topic}")
            return True
        
        try:
            response = messaging.unsubscribe_from_topic(device_tokens, topic)
            return response.failure_count == 0
        except Exception as e:
            logger.error(f"Topic unsubscription failed: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get notification statistics."""
        return {
            "initialized": self._initialized,
            "mock_mode": self._mock_mode,
            "sent_count": self._sent_count,
            "failed_count": self._failed_count
        }


# Global FCM service instance
fcm_service = FCMService()


# ============================================
# Convenience Functions
# ============================================

async def send_fall_alert(
    guardian_tokens: List[str],
    elderly_name: str,
    location: str = None
) -> Dict[str, Any]:
    """Send fall detection alert to guardians."""
    notification = PushNotification(
        title="🚨 Fall Detected!",
        body=f"{elderly_name} may have fallen. Please check immediately.",
        notification_type=NotificationType.FALL_DETECTED,
        data={
            "elderly_name": elderly_name,
            "location": location or "Unknown",
            "action": "view_alert"
        }
    )
    return await fcm_service.send_to_multiple(guardian_tokens, notification)


async def send_sos_alert(
    guardian_tokens: List[str],
    elderly_name: str
) -> Dict[str, Any]:
    """Send SOS button press alert to guardians."""
    notification = PushNotification(
        title="🆘 SOS Alert!",
        body=f"{elderly_name} pressed the SOS button and needs help!",
        notification_type=NotificationType.SOS_ALERT,
        data={
            "elderly_name": elderly_name,
            "action": "call_elderly"
        }
    )
    return await fcm_service.send_to_multiple(guardian_tokens, notification)


async def send_geofence_alert(
    guardian_tokens: List[str],
    elderly_name: str,
    zone_name: str
) -> Dict[str, Any]:
    """Send geofence breach alert to guardians."""
    notification = PushNotification(
        title="📍 Geofence Alert",
        body=f"{elderly_name} has left the '{zone_name}' safe zone.",
        notification_type=NotificationType.GEOFENCE_EXIT,
        data={
            "elderly_name": elderly_name,
            "zone_name": zone_name,
            "action": "view_location"
        }
    )
    return await fcm_service.send_to_multiple(guardian_tokens, notification)


async def send_hydration_reminder(
    device_token: str,
    current_ml: int,
    goal_ml: int
) -> Optional[str]:
    """Send hydration reminder to elderly user."""
    remaining = goal_ml - current_ml
    notification = PushNotification(
        title="💧 Hydration Reminder",
        body=f"You need {remaining}ml more water today. Stay hydrated!",
        notification_type=NotificationType.HYDRATION_REMINDER,
        data={
            "current_ml": str(current_ml),
            "goal_ml": str(goal_ml)
        }
    )
    return await fcm_service.send_to_device(device_token, notification)


async def send_exercise_reminder(
    device_token: str,
    exercise_name: str
) -> Optional[str]:
    """Send exercise reminder to elderly user."""
    notification = PushNotification(
        title="🏃 Exercise Time!",
        body=f"Time for your {exercise_name} session. Stay active!",
        notification_type=NotificationType.EXERCISE_REMINDER,
        data={
            "exercise_name": exercise_name,
            "action": "start_exercise"
        }
    )
    return await fcm_service.send_to_device(device_token, notification)
