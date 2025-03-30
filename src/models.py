import re
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, field_validator, HttpUrl, Field


class VideoDownloadConfig(BaseModel):
    url: HttpUrl
    video_format: str = "mp4"
    audio_format: str = "mp3"
    video_quality: str = "best"
    audio_quality: str = "320k"
    output_dir: Optional[Path] = None
    video_only: bool = False
    audio_only: bool = False
    quality: str = "high"
    format: Optional[str] = None
    downsize: bool = False
    max_size_mb: int = 8

    @field_validator("url")
    def validate_youtube_url(cls, v):
        # Regex patterns for different YouTube URL formats
        youtube_video_pattern = r'^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[a-zA-Z0-9_-]{11}.*$'
        youtube_playlist_pattern = r'^(https?://)?(www\.)?youtube\.com/playlist\?list=[a-zA-Z0-9_-]+.*$'
        
        # Convert to string in case it's passed as HttpUrl object
        url_str = str(v)
        
        # Check if it matches any of the valid YouTube URL patterns
        if re.match(youtube_video_pattern, url_str) or re.match(youtube_playlist_pattern, url_str):
            return v
            
        raise ValueError("URL must be a valid YouTube video or playlist link")

    @field_validator("quality")
    def validate_quality(cls, v):
        valid_qualities = ["high", "medium", "low"]
        if v.lower() not in valid_qualities:
            raise ValueError(f"Quality must be one of: {', '.join(valid_qualities)}")
        return v.lower()

    @field_validator("video_format", "audio_format")
    def sanitize_format(cls, v):
        # Sanitize format strings to prevent potential command injection
        # Only allow alphanumeric formats
        if not re.match(r'^[a-zA-Z0-9]+$', v):
            raise ValueError(f"Format must contain only letters and numbers")
        return v.lower()
