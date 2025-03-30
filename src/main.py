import os
import logging
import hashlib
import tempfile
import re
import argparse
import shutil
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager
import time

import yt_dlp
import ffmpeg
from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler
from rich.status import Status
from pydantic import BaseModel, ValidationError, field_validator, HttpUrl, Field


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

    @field_validator("url")
    def validate_youtube_url(cls, v):
        # More robust YouTube URL validation using regex
        youtube_pattern = r'^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[a-zA-Z0-9_-]{11}.*$'
        if not re.match(youtube_pattern, str(v)):
            raise ValueError("URL must be a valid YouTube video link")
        return v

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


class YouTubeDownloader:
    def __init__(self, config: VideoDownloadConfig, log_level: int = logging.INFO):
        # Set up working directory and final output directory
        self.output_dir = config.output_dir or Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store the config
        self.config = config

        # Set video quality parameters based on quality setting
        self._set_quality_parameters()

        # Set output format if specified
        if config.format:
            self.config.video_format = config.format

        # Set up console and logging
        self.console = Console()

        # Use a custom formatter that redacts sensitive information
        logging.basicConfig(
            level=log_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=self.console, rich_tracebacks=False)],  # Disable rich tracebacks for security
        )
        self.logger = logging.getLogger("youtube_downloader")

        # Rate limiting settings
        self.rate_limit_delay = 2  # seconds between operations

    @contextmanager
    def _temp_directory(self):
        """Create a temporary directory that auto-cleans using context manager"""
        temp_dir = tempfile.mkdtemp(prefix="youtube_downloader_")
        try:
            yield Path(temp_dir)
        finally:
            try:
                shutil.rmtree(temp_dir)
                self.logger.debug(f"Cleaned up temporary directory")
            except Exception as e:
                self.logger.warning(f"Failed to clean up temp directory: {str(e)}")

    def _set_quality_parameters(self):
        """Set quality parameters based on the quality setting"""
        quality_map = {
            "high": {
                "video_quality": "bestvideo+bestaudio/best",
                "audio_quality": "320k",
                "video_crf": 18,
                "video_preset": "slow"
            },
            "medium": {
                "video_quality": "bestvideo[height<=720]+bestaudio/best[height<=720]",
                "audio_quality": "192k",
                "video_crf": 23,
                "video_preset": "medium"
            },
            "low": {
                "video_quality": "bestvideo[height<=480]+bestaudio/best[height<=480]",
                "audio_quality": "128k",
                "video_crf": 28,
                "video_preset": "fast"
            }
        }

        # Get quality settings with fallback
        quality_settings = quality_map.get(self.config.quality, quality_map["high"])

        # Store settings as instance variables for later use
        self.config.video_quality = quality_settings["video_quality"]
        self.config.audio_quality = quality_settings["audio_quality"]
        self.video_crf = quality_settings["video_crf"]
        self.video_preset = quality_settings["video_preset"]

    def validate_url(self) -> bool:
        """Validate the provided URL"""
        try:
            # URL is already validated by Pydantic, just log it
            # Redact part of the URL in logs for privacy
            url_str = str(self.config.url)
            visible_part = url_str[:20]  # Show only beginning
            self.logger.info(f"URL Validated: {visible_part}...")
            return True
        except Exception as e:
            self.console.print(
                Panel(
                    f"URL Validation Error: Invalid URL format",
                    title="Validation Failed",
                    border_style="red",
                )
            )
            return False

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filenames to prevent path traversal and command injection"""
        # Replace potentially dangerous characters
        sanitized = re.sub(r'[^\w\-\.]', '_', filename)
        # Ensure the filename doesn't start with dots or dashes (which could be interpreted as options)
        sanitized = sanitized.lstrip('.-')
        return sanitized

    def download_video(self, temp_dir: Path) -> Optional[Path]:
        """Download video from YouTube URL"""
        # Create a custom progress hook
        progress_status = self.console.status("[bold blue]Downloading video...", spinner="dots")
        progress_status.start()

        try:
            # Create a callback to properly handle progress updates
            def progress_hook(d):
                self._download_progress(d, progress_status)

            # Setup yt-dlp options with secure defaults
            ydl_opts = {
                "format": self.config.video_quality,
                "outtmpl": str(temp_dir / "%(title)s.%(ext)s"),
                "merge_output_format": self.config.video_format,
                "quiet": False,  # Change to False for more detailed logging
                "no_warnings": False,  # Change to False to see warnings
                "progress_hooks": [progress_hook],
                # Add rate limiting
                "sleep_interval": 1,  # Sleep between requests
                "max_sleep_interval": 5,
                # Enhanced error handling
                "ignoreerrors": False,
                "no_color": True,
            }

            # Adjust options for audio-only mode
            if self.config.audio_only:
                ydl_opts.update({
                    "format": "bestaudio/best",
                    "postprocessors": [{
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": self.config.audio_format,
                        "preferredquality": self.config.audio_quality.rstrip('k')
                    }],
                    # Ensure audio extraction
                    "writethumbnail": False,
                    "nooverwrites": True,
                })

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    info_dict = ydl.extract_info(str(self.config.url), download=True)
                    progress_status.stop()

                    # Find the downloaded file
                    possible_extensions = [
                        self.config.audio_format if self.config.audio_only else self.config.video_format, 
                        'mp3', 'mp4', 'mkv', 'webm', 'wav', 'aac', 'flac'
                    ]

                    for ext in possible_extensions:
                        potential_files = list(temp_dir.glob(f"*.{ext}"))
                        if potential_files:
                            downloaded_file = potential_files[0]
                            # Sanitize filename before returning
                            safe_filename = self._sanitize_filename(downloaded_file.name)
                            safe_path = temp_dir / safe_filename

                            # If the sanitized name is different, rename the file
                            if safe_filename != downloaded_file.name:
                                downloaded_file.rename(safe_path)
                                downloaded_file = safe_path

                            self.logger.info(f"{'Audio' if self.config.audio_only else 'Video'} Downloaded: {safe_filename}")
                            return downloaded_file

                    # Detailed logging if no file is found
                    self.logger.error(f"No files found with extensions: {possible_extensions}")
                    self.logger.error(f"Temp directory contents: {list(temp_dir.iterdir())}")
                    return None

                except Exception as extract_error:
                    # More detailed error logging
                    error_msg = str(extract_error)
                    self.logger.error(f"Extraction Error: {error_msg}")
                    
                    # Print more context about the error
                    self.console.print(
                        Panel(
                            f"Detailed Download Error: {error_msg}",
                            title="Download Failed",
                            border_style="red"
                        )
                    )
                    return None

        except Exception as e:
            progress_status.stop()
            # Avoid logging the full exception which might contain sensitive information
            error_msg = str(e)
            # Truncate long error messages
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."

            self.console.print(
                Panel(
                    f"Download Error: {error_msg}",
                    title="Download Failed",
                    border_style="red"
                )
            )
            self.logger.error(f"Download error: {error_msg}")
            return None

    def _download_progress(self, d, status):
        """Handle download progress updates"""
        if d['status'] == 'downloading':
            percent = d.get('_percent_str', 'N/A')
            speed = d.get('_speed_str', 'N/A')
            eta = d.get('_eta_str', 'N/A')
            status.update(f"[bold blue]Downloading: {percent} at {speed}, ETA: {eta}")
        elif d['status'] == 'finished':
            status.update("[bold green]Download complete, processing...")

    def optimize_video(self, input_path: Path, temp_dir: Path) -> Optional[Path]:
        """Optimize video using FFmpeg with security considerations and enhanced error handling"""
        if not input_path.exists():
            self.logger.error(f"Input file does not exist: {input_path.name}")
            return None

        try:
            # Create a safe output filename
            safe_filename = self._sanitize_filename(f"{input_path.stem}_optimized.{self.config.video_format}")
            output_path = temp_dir / safe_filename

            # Get input video metadata to estimate processing time
            try:
                probe = ffmpeg.probe(str(input_path))
                duration = float(probe['streams'][0].get('duration', 0))
            except Exception as probe_error:
                self.logger.warning(f"Could not probe video duration: {probe_error}")
                duration = 600  # Default to 10 minutes if probe fails

            # Estimate a reasonable timeout based on video duration
            # Add 5 minutes of buffer time for longer videos
            timeout = max(300, int(duration * 2))

            with self.console.status(f"[bold blue]Optimizing video ({self.config.quality} quality)...") as status:
                # Enhanced FFmpeg parameters for robust optimization
                input_stream = ffmpeg.input(str(input_path))
                
                # Dynamically choose video codec based on output format
                video_codec = "libx264"  # Default
                if self.config.video_format in ["mkv", "webm"]:
                    video_codec = "libvpx-vp9" if self.config.video_format == "webm" else "libx264"

                output_stream = (
                    input_stream
                    .output(
                        str(output_path),
                        vcodec=video_codec,
                        crf=self.video_crf,
                        preset=self.video_preset,
                        # Additional optimization parameters
                        movflags="+faststart",  # Web optimization
                        acodec="aac",  # Standard audio codec
                        audio_bitrate="128k"  # Default audio bitrate
                    )
                    .global_args('-loglevel', 'error')  # More error visibility
                    .overwrite_output()
                )

                # Run with more comprehensive error handling and timeout
                try:
                    import subprocess
                    import threading
                    import queue

                    # Prepare FFmpeg command
                    ffmpeg_command = output_stream.compile()

                    # Queue to store output
                    output_queue = queue.Queue()

                    # Function to read output
                    def enqueue_output(out, queue):
                        for line in iter(out.readline, b''):
                            queue.put(line)
                        out.close()

                    # Start the process
                    process = subprocess.Popen(
                        ffmpeg_command, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        universal_newlines=True
                    )

                    # Thread to read output
                    output_thread = threading.Thread(
                        target=enqueue_output, 
                        args=(process.stderr, output_queue)
                    )
                    output_thread.daemon = True
                    output_thread.start()

                    # Wait for process with timeout
                    try:
                        # Use poll with a timeout mechanism
                        start_time = time.time()
                        while process.poll() is None:
                            # Check for timeout
                            if time.time() - start_time > timeout:
                                process.kill()
                                self.logger.error(f"Video optimization timed out after {timeout} seconds")
                                return None
                            
                            # Small sleep to prevent busy waiting
                            time.sleep(1)

                        # Collect any remaining error output
                        error_output = []
                        while not output_queue.empty():
                            error_output.append(output_queue.get())

                        # Check return code
                        if process.returncode != 0:
                            error_msg = ''.join(error_output)
                            self.logger.error(f"FFmpeg optimization failed: {error_msg}")
                            return None

                    except Exception as process_error:
                        process.kill()
                        self.logger.error(f"Process error during optimization: {process_error}")
                        return None

                except Exception as e:
                    # Log detailed FFmpeg error for debugging
                    error_details = f"FFmpeg Optimization Error: {str(e)}"
                    self.logger.error(error_details)
                    return None

            # Verify output file was created
            if not output_path.exists() or output_path.stat().st_size == 0:
                self.logger.error(f"Optimization failed: Output file is empty or not created")
                return None

            self.logger.info(f"Video Optimized: {output_path.name}")
            return output_path

        except Exception as e:
            self.console.print(
                Panel(
                    f"Video Optimization Error: {str(e)}",
                    title="Optimization Failed",
                    border_style="red",
                )
            )
            self.logger.error(f"Detailed optimization error: {str(e)}")
            return None

    def extract_audio(self, input_path: Path, temp_dir: Path) -> Optional[Path]:
        """Enhanced audio extraction with robust codec and stream handling"""
        if not input_path.exists():
            self.logger.error(f"Input file does not exist: {input_path.name}")
            return None

        # Advanced codec and stream mapping
        codec_map = {
            "mp3": {"codec": "libmp3lame", "extension": "mp3"},
            "aac": {"codec": "aac", "extension": "aac"},
            "ogg": {"codec": "libvorbis", "extension": "ogg"},
            "flac": {"codec": "flac", "extension": "flac"},
            "wav": {"codec": "pcm_s16le", "extension": "wav"}
        }

        # Validate audio format
        audio_format = self.config.audio_format.lower()
        if audio_format not in codec_map:
            self.logger.error(f"Unsupported audio format: {audio_format}")
            self.console.print(
                Panel(
                    f"Unsupported Audio Format: {audio_format}. Supported formats are: {', '.join(codec_map.keys())}",
                    title="Format Error",
                    border_style="red"
                )
            )
            return None

        # Probe input file to get audio stream information
        try:
            import ffmpeg
            probe_result = ffmpeg.probe(str(input_path))
            audio_streams = [
                stream for stream in probe_result['streams'] 
                if stream['codec_type'] == 'audio'
            ]

            # Log stream information for debugging
            self.logger.info(f"Found {len(audio_streams)} audio streams")
            for i, stream in enumerate(audio_streams):
                self.logger.info(f"Stream {i}: {stream.get('codec_name', 'Unknown')} - {stream.get('channels', 'N/A')} channels")

            # Select best audio stream (most channels, highest bitrate)
            if not audio_streams:
                self.logger.error("No audio streams found in the input file")
                self.console.print(
                    Panel(
                        "No audio streams found in the input file",
                        title="Extraction Error",
                        border_style="red"
                    )
                )
                return None

            selected_stream = max(
                audio_streams, 
                key=lambda s: (
                    int(s.get('channels', 0)), 
                    int(s.get('bit_rate', 0) or 0)
                )
            )
            selected_stream_index = audio_streams.index(selected_stream)

        except Exception as probe_error:
            self.logger.error(f"Failed to probe audio streams: {probe_error}")
            self.console.print(
                Panel(
                    f"Audio Stream Probe Error: {probe_error}",
                    title="Probe Failed",
                    border_style="red"
                )
            )
            selected_stream_index = 0  # Default to first stream

        # Unique filename generation
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        safe_filename = self._sanitize_filename(
            f"{input_path.stem}_{unique_id}.{codec_map[audio_format]['extension']}"
        )
        audio_path = temp_dir / safe_filename

        # FFmpeg command with advanced stream selection
        ffmpeg_command = [
            'ffmpeg',
            '-i', str(input_path),
            '-map', f'0:a:{selected_stream_index}',  # Explicitly select stream
            '-vn',  # No video
            '-acodec', codec_map[audio_format]['codec'],
            '-b:a', self.config.audio_quality,
            '-y',  # Overwrite
            str(audio_path)
        ]

        try:
            # Run FFmpeg with error capture
            import subprocess
            process = subprocess.Popen(
                ffmpeg_command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            # Capture output
            stdout, stderr = process.communicate(timeout=300)  # 5-minute timeout

            # Check return code
            if process.returncode != 0:
                # Log detailed error
                error_details = f"""
FFmpeg Audio Extraction Error:
Command: {' '.join(ffmpeg_command)}
Stdout: {stdout}
Stderr: {stderr}
Selected Stream: {selected_stream_index}
Stream Details: {selected_stream}
"""
                self.logger.error(error_details)
                
                # Print error to console with detailed panel
                self.console.print(
                    Panel(
                        f"Audio Extraction Failed:\n{stderr}",
                        title="Extraction Error",
                        border_style="red",
                        expand=False
                    )
                )
                return None

        except subprocess.TimeoutExpired:
            # Handle timeout
            self.logger.error("Audio extraction timed out")
            self.console.print(
                Panel(
                    "Audio extraction process timed out",
                    title="Timeout Error",
                    border_style="red"
                )
            )
            return None
        except Exception as e:
            # Catch any other unexpected errors
            self.logger.error(f"Unexpected audio extraction error: {str(e)}")
            self.console.print(
                Panel(
                    f"Unexpected Audio Extraction Error: {str(e)}",
                    title="Extraction Failed",
                    border_style="red"
                )
            )
            return None

        # Verify output file was created
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            self.logger.error(f"Audio extraction failed: Output file is empty or not created")
            self.console.print(
                Panel(
                    "Audio extraction resulted in an empty file",
                    title="Extraction Failed",
                    border_style="red"
                )
            )
            return None

        self.logger.info(f"Audio Extracted: {audio_path.name}")
        return audio_path

    def hash_file(self, file_path: Path) -> str:
        """Create a SHA-256 hash of the file"""
        if not file_path or not file_path.exists():
            return "File not available"

        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.error(f"Error hashing file: {file_path.name}")
            return "Hash calculation failed"

    def verify_hash(self, file_path: Path, expected_hash: str) -> bool:
        """Verify a file's hash matches the expected value"""
        if not file_path.exists():
            return False

        actual_hash = self.hash_file(file_path)
        return actual_hash == expected_hash

    def copy_to_output_dir(self, file_path: Path) -> Optional[Path]:
        """Copy file from temp directory to output directory with security checks"""
        if not file_path or not file_path.exists():
            return None

        try:
            # Create a safe filename
            safe_filename = self._sanitize_filename(file_path.name)
            destination = self.output_dir / safe_filename

            # Check if destination exists and is outside of our output directory (path traversal check)
            destination = destination.resolve()
            if not str(destination).startswith(str(self.output_dir.resolve())):
                self.logger.error(f"Security error: Attempted path traversal detected")
                return None

            # Copy the file
            shutil.copy2(file_path, destination)
            self.logger.info(f"File copied to output directory")
            return destination
        except Exception as e:
            self.logger.error(f"Failed to copy file")
            return None

    def process_video(self) -> Dict[str, Any]:
        """Process video from URL to optimized video and audio files using context manager"""
        result = {
            "success": False,
            "video_path": None,
            "audio_path": None,
            "error": None
        }

        # Validate URL
        if not self.validate_url():
            result["error"] = "URL validation failed"
            return result

        # Use context manager for temporary directory
        with self._temp_directory() as temp_dir:
            # Download video
            downloaded_video = self.download_video(temp_dir)
            if not downloaded_video:
                result["error"] = "Video download failed"
                return result

            # Process according to mode
            try:
                # Handle video processing if needed
                if not self.config.audio_only:
                    optimized_video = self.optimize_video(downloaded_video, temp_dir)
                    if optimized_video:
                        final_video_path = self.copy_to_output_dir(optimized_video)
                        result["video_path"] = final_video_path

                        # Verify hash after copy to ensure integrity
                        original_hash = self.hash_file(optimized_video)
                        if not self.verify_hash(final_video_path, original_hash):
                            self.logger.warning("Video hash verification failed - possible corruption")

                # Handle audio processing if needed
                if not self.config.video_only:
                    audio_file = self.extract_audio(downloaded_video, temp_dir)
                    if audio_file:
                        final_audio_path = self.copy_to_output_dir(audio_file)
                        result["audio_path"] = final_audio_path

                        # Verify hash after copy
                        original_hash = self.hash_file(audio_file)
                        if not self.verify_hash(final_audio_path, original_hash):
                            self.logger.warning("Audio hash verification failed - possible corruption")

                # Success if either video or audio was processed successfully
                result["success"] = bool(result["video_path"] or result["audio_path"])

                # Log details if successful
                if result["success"]:
                    # Create log details with minimal sensitive information
                    log_details = {
                        "timestamp": self._get_current_timestamp(),
                        "video_processed": bool(result["video_path"]),
                        "audio_processed": bool(result["audio_path"]),
                        "video_format": self.config.video_format,
                        "audio_format": self.config.audio_format,
                        "quality_setting": self.config.quality,
                        "video_hash": self.hash_file(result["video_path"]) if result["video_path"] else None,
                        "audio_hash": self.hash_file(result["audio_path"]) if result["audio_path"] else None,
                    }
                    self._log_details(log_details)

                return result

            except Exception as e:
                # Generic error message without sensitive details
                result["error"] = "Processing failed"
                self.logger.error(f"Processing error: {str(e)}")
                return result

    def _get_current_timestamp(self):
        """Get a formatted timestamp for logging"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _log_details(self, details: Dict[str, Any]):
        """Log download and processing details to a JSON file with security considerations"""
        try:
            log_file = self.output_dir / "download_log.json"

            # Ensure we're not writing outside our intended directory
            log_file = log_file.resolve()
            if not str(log_file).startswith(str(self.output_dir.resolve())):
                self.logger.error("Security error: Attempted path traversal in log file path")
                return

            with open(log_file, "w") as f:
                json.dump(details, f, indent=4)
            self.logger.info(f"Log saved")
        except Exception as e:
            self.logger.error(f"Failed to write log file")


def parse_arguments():
    """Parse command line arguments with input validation"""
    parser = argparse.ArgumentParser(
        description="YouTube Video/Audio Downloader & Optimizer"
    )
    parser.add_argument(
        "url", nargs="?", help="YouTube URL to download (if not provided, will prompt)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        help="Output directory for downloaded files"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "-q", "--quality",
        type=str,
        choices=["high", "medium", "low"],
        default="high",
        help="Quality preset (default: high)"
    )
    parser.add_argument(
        "-f", "--format",
        type=str,
        help="Output video format (e.g., mp4, mkv, webm)"
    )
    parser.add_argument(
        "--video-format",
        type=str,
        default="mp4",
        help="Video format (default: mp4)"
    )
    parser.add_argument(
        "--audio-format",
        type=str,
        default="mp3",
        choices=["mp3", "aac", "ogg", "flac"],
        help="Audio format (default: mp3)"
    )
    parser.add_argument(
        "--audio-quality",
        type=str,
        default="320k",
        help="Audio quality bitrate (default: 320k)"
    )
    parser.add_argument(
        "--video-only",
        action="store_true",
        help="Only download and optimize video"
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Only download and extract audio"
    )

    args = parser.parse_args()

    # Additional validation of output directory
    if args.output_dir:
        output_path = Path(args.output_dir)
        # Resolve to absolute path to prevent path traversal
        output_path = output_path.resolve()
        # Convert back to string for the args
        args.output_dir = str(output_path)

    return args


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Set up console
    console = Console()
    console.print(
        Panel("YouTube Video Downloader", title="🎥 Welcome", border_style="blue")
    )

    # Determine log level from arguments
    log_level = logging.WARNING if args.quiet else logging.INFO

    # Get URL from arguments or prompt with validation
    url = args.url
    if not url:
        url = console.input("Enter YouTube Video URL: ")
        # Basic validation at input time
        youtube_pattern = r'^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[a-zA-Z0-9_-]{11}.*$'
        if not re.match(youtube_pattern, url):
            console.print(Panel("Invalid YouTube URL format", title="Error", border_style="red"))
            return

    # Set up output directory with security validation
    if args.output_dir:
        output_dir = Path(args.output_dir)
        # Resolve to absolute path and validate
        output_dir = output_dir.resolve()
    else:
        output_dir = Path.cwd()

    # Create configuration
    try:
        # Sanitize format inputs before passing to config
        video_format = args.format or args.video_format
        audio_format = args.audio_format

        # Simple format validation
        format_pattern = r'^[a-zA-Z0-9]+$'
        if video_format and not re.match(format_pattern, video_format):
            console.print(Panel("Invalid video format - use only letters and numbers", title="Error", border_style="red"))
            return

        if audio_format and not re.match(format_pattern, audio_format):
            console.print(Panel("Invalid audio format - use only letters and numbers", title="Error", border_style="red"))
            return

        config = VideoDownloadConfig(
            url=url,
            video_format=video_format,
            audio_format=audio_format,
            audio_quality=args.audio_quality,
            output_dir=output_dir,
            video_only=args.video_only,
            audio_only=args.audio_only,
            quality=args.quality,
            format=args.format
        )
    except ValidationError as e:
        # Sanitize error message to avoid exposing details
        console.print(
            Panel(
                "Configuration validation failed. Please check your inputs.",
                title="Validation Failed",
                border_style="red",
            )
        )
        return

    # Initialize downloader with configuration
    downloader = YouTubeDownloader(config=config, log_level=log_level)

    # Process the video
    result = downloader.process_video()

    # Display results
    if result["success"]:
        # Build status message
        video_path = result['video_path'].name if result['video_path'] else "Not processed"
        audio_path = result['audio_path'].name if result['audio_path'] else "Not processed"

        if not result['video_path'] and not result['audio_path']:
            console.print(
                Panel(
                    "Process completed, but no files were produced.",
                    title="Warning",
                    border_style="yellow",
                )
            )
        else:
            console.print(
                Panel(
                    f"[green]Process completed successfully![/green]\n\n"
                    f"Video: {video_path}\n"
                    f"Audio: {audio_path}\n\n"
                    f"Quality: {config.quality}\n"
                    f"Files are stored in: {output_dir}",
                    title="Download Complete",
                    border_style="green",
                )
            )
    else:
        console.print(
            Panel(
                f"Process failed: {result['error']}",
                title="Download Failed",
                border_style="red",
            )
        )


if __name__ == "__main__":
    main()
