import re
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def sanitize_filename(filename: str) -> str:
    """Sanitize filenames to prevent path traversal and command injection"""
    # Replace potentially dangerous characters
    sanitized = re.sub(r'[^\w\-\.]', '_', filename)
    # Ensure the filename doesn't start with dots or dashes (which could be interpreted as options)
    sanitized = sanitized.lstrip('.-')
    return sanitized


def validate_youtube_url(url: str) -> bool:
    """Validate that a string is a YouTube URL"""
    youtube_pattern = r'^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[a-zA-Z0-9_-]{11}.*$'
    return bool(re.match(youtube_pattern, url))


def hash_file(file_path: Path) -> Optional[str]:
    """Create a SHA-256 hash of the file"""
    if not file_path or not file_path.exists():
        return None

    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception:
        return None


def verify_hash(file_path: Path, expected_hash: str) -> bool:
    """Verify a file's hash matches the expected value"""
    if not file_path.exists():
        return False

    actual_hash = hash_file(file_path)
    if not actual_hash:
        return False
        
    return actual_hash == expected_hash


def get_current_timestamp() -> str:
    """Get a formatted timestamp for logging"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_log_details(
    video_processed: bool, 
    audio_processed: bool,
    video_format: str,
    audio_format: str,
    quality_setting: str,
    video_path: Optional[Path] = None,
    audio_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Create log details with minimal sensitive information"""
    return {
        "timestamp": get_current_timestamp(),
        "video_processed": video_processed,
        "audio_processed": audio_processed,
        "video_format": video_format,
        "audio_format": audio_format,
        "quality_setting": quality_setting,
        "video_hash": hash_file(video_path) if video_path else None,
        "audio_hash": hash_file(audio_path) if audio_path else None,
    }
