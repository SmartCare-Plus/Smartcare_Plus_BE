"""
SMARTCARE+ accessibility APIs.

Speech-to-text is proxied through the backend so Gemini credentials are never
shipped in the mobile app.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from google.auth.transport.requests import Request
from google.oauth2 import service_account

from core.auth import AuthenticatedUser, verify_firebase_token
from core.config import settings

router = APIRouter()
logger = logging.getLogger("smartcare.accessibility")

_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def _extract_text(payload: dict[str, Any]) -> str:
    parts = (
        payload.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [])
    )
    text_parts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    return " ".join(part.strip() for part in text_parts if part.strip()).strip()


async def _transcribe_with_gemini(
    audio_bytes: bytes,
    mime_type: str,
    locale: str,
) -> str:
    if not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Gemini STT is not configured. Set GEMINI_API_KEY on the backend.",
        )

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.GEMINI_STT_MODEL}:generateContent"
    )
    prompt = (
        "Transcribe the speech in this audio exactly for a text input field. "
        f"The expected language is {locale}. Return only the transcript text. "
        "If there is no clear speech, return an empty string."
    )
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(audio_bytes).decode("ascii"),
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 512,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(
                endpoint,
                headers={
                    "x-goog-api-key": settings.GEMINI_API_KEY,
                    "Content-Type": "application/json",
                },
                json=body,
            )
    except httpx.HTTPError as exc:
        logger.warning("Gemini STT request failed: %s", exc)
        raise HTTPException(status_code=502, detail="Gemini STT request failed") from exc

    if response.status_code >= 400:
        logger.warning(
            "Gemini STT returned %s: %s",
            response.status_code,
            response.text[:500],
        )
        raise HTTPException(status_code=502, detail="Gemini STT failed")

    transcript = _extract_text(response.json())
    return transcript.strip().strip('"').strip()


def _resolve_vertex_credentials_path() -> Path:
    if settings.VERTEX_AI_CREDENTIALS_PATH:
        path = Path(settings.VERTEX_AI_CREDENTIALS_PATH)
        if not path.is_absolute():
            path = Path(__file__).parent.parent / path
        return path
    return Path(__file__).parent.parent / settings.FIREBASE_CREDENTIALS_PATH


def _load_vertex_credentials():
    credentials_path = _resolve_vertex_credentials_path()
    if not credentials_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Vertex AI credentials file not found: {credentials_path}",
        )

    credentials = service_account.Credentials.from_service_account_file(
        str(credentials_path),
        scopes=[_CLOUD_PLATFORM_SCOPE],
    )
    project_id = settings.VERTEX_AI_PROJECT_ID or credentials.project_id
    if not project_id:
        raise HTTPException(
            status_code=500,
            detail="Vertex AI project ID is not configured.",
        )
    credentials.refresh(Request())
    return credentials, project_id


async def _transcribe_with_vertex_ai(
    audio_bytes: bytes,
    mime_type: str,
    locale: str,
) -> str:
    credentials, project_id = _load_vertex_credentials()
    location = settings.VERTEX_AI_LOCATION
    endpoint = (
        f"https://{location}-aiplatform.googleapis.com/v1/"
        f"projects/{project_id}/locations/{location}/publishers/google/models/"
        f"{settings.GEMINI_STT_MODEL}:generateContent"
    )
    prompt = (
        "Transcribe the speech in this audio exactly for a text input field. "
        f"The expected language is {locale}. Return only the transcript text. "
        "If there is no clear speech, return an empty string."
    )
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64.b64encode(audio_bytes).decode("ascii"),
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 512,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {credentials.token}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
    except httpx.HTTPError as exc:
        logger.warning("Vertex AI STT request failed: %s", exc)
        raise HTTPException(status_code=502, detail="Vertex AI STT request failed") from exc

    if response.status_code >= 400:
        logger.warning(
            "Vertex AI STT returned %s: %s",
            response.status_code,
            response.text[:500],
        )
        raise HTTPException(status_code=502, detail="Vertex AI STT failed")

    transcript = _extract_text(response.json())
    return transcript.strip().strip('"').strip()


async def _transcribe_audio(
    audio_bytes: bytes,
    mime_type: str,
    locale: str,
) -> tuple[str, str]:
    provider = settings.STT_PROVIDER.strip().lower()
    if provider == "vertex_ai":
        return await _transcribe_with_vertex_ai(audio_bytes, mime_type, locale), "vertex_ai"
    if provider == "gemini_api":
        return await _transcribe_with_gemini(audio_bytes, mime_type, locale), "gemini_api"
    raise HTTPException(
        status_code=500,
        detail=f"Unsupported STT_PROVIDER: {settings.STT_PROVIDER}",
    )


@router.post("/speech-to-text")
async def speech_to_text(
    audio: UploadFile = File(...),
    locale: str = Form(default=settings.STT_DEFAULT_LOCALE),
    current_user: AuthenticatedUser = Depends(verify_firebase_token),
):
    """Transcribe a short voice clip for accessibility text input."""
    content_type = audio.content_type or "audio/wav"
    if content_type == "application/octet-stream":
        content_type = "audio/wav"
    if not content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid audio upload")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    max_bytes = settings.STT_MAX_AUDIO_MB * 1024 * 1024
    if len(audio_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large. Max {settings.STT_MAX_AUDIO_MB}MB allowed.",
        )

    text, provider = await _transcribe_audio(audio_bytes, content_type, locale)
    logger.info(
        "STT completed for uid=%s locale=%s bytes=%s",
        current_user.uid,
        locale,
        len(audio_bytes),
    )
    return {
        "text": text,
        "provider": provider,
        "model": settings.GEMINI_STT_MODEL,
        "locale": locale,
    }
