import os
import logging
import hashlib
import tempfile
import re
import shutil
import json
import time
import uuid
import subprocess
import threading
import queue
import concurrent.futures
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager

import yt_dlp
import ffmpeg
from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler
from rich.status import Status

from models import VideoDownloadConfig


class YouTubeDownloader:
    def __init__(self, config: VideoDownloadConfig, log_level: int = logging.INFO):
        # Create a UUID for this run
        self.run_uuid = str(uuid.uuid4())[:8]
        
        # Set up logging
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
            force=True
        )
        self.logger = logging.getLogger(f"downloader_{self.run_uuid}")
        
        # Set up working directory and final output directory
        base_dir = config.output_dir or Path.cwd()
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a dedicated output folder with UUID
        self.output_dir = base_dir / f"{self.run_uuid}_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a metadata file to track processing information
        self.metadata_path = self.output_dir / "metadata.json"

        # Store the config
        self.config = config
        
        # Initialize video cache
        self.cache_dir = base_dir / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index = self.cache_dir / "cache_index.json"
        self._init_cache()
        
        # Detect hardware acceleration
        self.hw_accel = self._detect_hardware_acceleration()
        
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
        self.logger.info(f"Created output directory: {self.output_dir}")

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
                
    def _init_cache(self):
        """Initialize the video cache system"""
        if not self.cache_index.exists():
            with open(self.cache_index, 'w') as f:
                json.dump({}, f)
                
    def _get_from_cache(self, video_id: str, quality: str, format: str) -> Optional[Path]:
        """Check if a video exists in cache and return its path"""
        try:
            with open(self.cache_index, 'r') as f:
                cache = json.load(f)
                
            cache_key = f"{video_id}_{quality}_{format}"
            if cache_key in cache:
                cached_file = Path(cache[cache_key]['path'])
                if cached_file.exists():
                    self.logger.info(f"Found video in cache: {cached_file.name}")
                    return cached_file
                else:
                    # Remove invalid cache entry
                    del cache[cache_key]
                    with open(self.cache_index, 'w') as f:
                        json.dump(cache, f)
        except Exception as e:
            self.logger.warning(f"Cache read error: {str(e)}")
            
        return None
        
    def _add_to_cache(self, video_id: str, quality: str, format: str, file_path: Path) -> None:
        """Add a video to the cache"""
        try:
            with open(self.cache_index, 'r') as f:
                cache = json.load(f)
                
            # Generate cache key
            cache_key = f"{video_id}_{quality}_{format}"
            
            # Copy file to cache directory
            cache_file = self.cache_dir / file_path.name
            shutil.copy2(file_path, cache_file)
            
            # Update cache index
            cache[cache_key] = {
                'path': str(cache_file),
                'created': time.time(),
                'size': cache_file.stat().st_size
            }
            
            with open(self.cache_index, 'w') as f:
                json.dump(cache, f)
                
            self.logger.info(f"Added video to cache: {cache_file.name}")
        except Exception as e:
            self.logger.warning(f"Cache write error: {str(e)}")
            
    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/.*?/([a-zA-Z0-9_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
                
        return None
        
    def _detect_hardware_acceleration(self) -> Optional[str]:
        """Detect available hardware acceleration"""
        try:
            # Try to detect available hardware acceleration
            hw_check = subprocess.run(
                ['ffmpeg', '-hide_banner', '-hwaccels'],
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                timeout=5,
                text=True
            )
            output = hw_check.stdout + hw_check.stderr
            
            # Check for various hardware acceleration options
            if 'cuda' in output or 'nvenc' in output:
                self.logger.info("Detected NVIDIA GPU acceleration")
                return 'cuda'
            elif 'videotoolbox' in output:  # macOS
                self.logger.info("Detected macOS VideoToolbox acceleration")
                return 'videotoolbox'
            elif 'qsv' in output:  # Intel QuickSync
                self.logger.info("Detected Intel QuickSync acceleration")
                return 'qsv'
            elif 'vaapi' in output:  # VA-API
                self.logger.info("Detected VA-API acceleration")
                return 'vaapi'
            else:
                self.logger.info("No hardware acceleration detected, using software encoding")
                return None
        except Exception as e:
            self.logger.warning(f"Hardware acceleration detection failed: {str(e)}")
            return None
            
    def _monitor_ffmpeg_progress(self, process, duration: float, status: Status) -> None:
        """Monitor FFmpeg progress and update status"""
        pattern = re.compile(r"time=(\d+):(\d+):(\d+\.?\d*)")
        
        for line in iter(process.stderr.readline, ''):
            match = pattern.search(line)
            if match:
                h, m, s = map(float, match.groups())
                current_seconds = h * 3600 + m * 60 + s
                progress = min(100, int(current_seconds / duration * 100))
                status.update(f"[bold blue]Downsizing video to {self.config.max_size_mb}MB... {progress}% complete")

    def _set_quality_parameters(self):
        """Set quality parameters based on the quality setting and adjust for size limits"""
        # Define format selection priorities based on quality and downsize preferences
        if self.config.downsize:
            # If downsizing is enabled, use more restrictive quality settings
            size_restricted_formats = {
                "high": {
                    "video_quality": "bestvideo[height<=1080][filesize_approx<={}M]+bestaudio[acodec=opus]/bestvideo[height<=1080]+bestaudio[acodec=opus]".format(max(5, self.config.max_size_mb - 3)),
                    "audio_quality": "256k",
                    "video_crf": 23,
                    "video_preset": "medium"
                },
                "medium": {
                    "video_quality": "bestvideo[height<=720][filesize_approx<={}M]+bestaudio[acodec=opus]/bestvideo[height<=720]+bestaudio".format(max(3, self.config.max_size_mb - 2)),
                    "audio_quality": "192k",
                    "video_crf": 26,
                    "video_preset": "medium"
                },
                "low": {
                    "video_quality": "bestvideo[height<=480][filesize_approx<={}M]+bestaudio[acodec=aac]/bestvideo[height<=480]+bestaudio".format(max(2, self.config.max_size_mb - 1)),
                    "audio_quality": "128k",
                    "video_crf": 30,
                    "video_preset": "fast"
                }
            }
            quality_map = size_restricted_formats
        else:
            # Standard quality presets with preferred codecs
            quality_map = {
                "high": {
                    "video_quality": "bestvideo[vcodec^=av01]+bestaudio[acodec=opus]/bestvideo[vcodec^=vp9]+bestaudio[acodec=opus]/bestvideo+bestaudio",
                    "audio_quality": "320k",
                    "video_crf": 18,
                    "video_preset": "slow"
                },
                "medium": {
                    "video_quality": "bestvideo[height<=720][vcodec^=vp9]+bestaudio[acodec=opus]/bestvideo[height<=720]+bestaudio",
                    "audio_quality": "192k",
                    "video_crf": 23,
                    "video_preset": "medium"
                },
                "low": {
                    "video_quality": "bestvideo[height<=480][vcodec^=avc]+bestaudio[acodec=aac]/bestvideo[height<=480]+bestaudio",
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
        """Download video from YouTube URL with enhanced format selection"""
        # Create a custom progress hook
        progress_status = self.console.status("[bold blue]Downloading video...", spinner="dots")
        progress_status.start()

        try:
            # Check if video exists in cache
            video_id = self._extract_video_id(str(self.config.url))
            if video_id and not self.config.downsize:  # Only use cache for non-downsized videos
                cached_path = self._get_from_cache(
                    video_id, 
                    self.config.quality, 
                    self.config.video_format
                )
                if cached_path:
                    progress_status.stop()
                    self.logger.info(f"Using cached video: {cached_path.name}")
                    # Copy cached file to output directory
                    output_file = self.output_dir / cached_path.name
                    shutil.copy2(cached_path, output_file)
                    return output_file
            # Create a callback to properly handle progress updates
            def progress_hook(d):
                self._download_progress(d, progress_status)

            # Setup yt-dlp options with secure defaults and improved codec selection
            ydl_opts = {
                # Use better format selection with codec preferences
                "format": self.config.video_quality + "/best",
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
                # Performance tuning
                "concurrent_fragments": 3,  # Use multiple connections for fragments
                # Add metadata
                "add_metadata": True,
                # Extra security and performance settings
                "socket_timeout": 30,  # Shorter socket timeout
                "retries": 3,  # Retry on connection errors
                "skip_unavailable_fragments": True,  # Skip unavailable fragments
                "geo_bypass": True,  # Try to bypass geo restrictions
                # Prefer better codecs
                "postprocessor_args": {
                    "ffmpeg": ["-threads", "4", "-nostdin"]
                },
            }

            # Adjust options for audio-only mode
            if self.config.audio_only:
                ydl_opts.update({
                    "format": "bestaudio[acodec=opus]/bestaudio/best",
                    "postprocessors": [{
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": self.config.audio_format,
                        "preferredquality": self.config.audio_quality.rstrip('k')
                    }, {
                        # Add metadata processor
                        "key": "FFmpegMetadata",
                        "add_metadata": True,
                    }],
                    # Ensure audio extraction
                    "writethumbnail": True,  # Download thumbnail
                    "nooverwrites": True,
                    "embed_thumbnail": True,  # Embed thumbnail in audio file if possible
                    # Prefer better audio codecs
                    "format_sort": [
                        "acodec:opus", "acodec:aac", "acodec:mp3", "acodec:vorbis", "acodec:flac"
                    ],
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
                            
                            # Add to cache if not in downsize mode and we have a video_id
                            if video_id and not self.config.downsize and not self.config.audio_only:
                                output_file = self.output_dir / safe_filename
                                shutil.copy2(safe_path, output_file)
                                self._add_to_cache(video_id, self.config.quality, self.config.video_format, output_file)
                                
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
        """Optimize video using FFmpeg only if downsize option is enabled and file is larger than max_size_mb"""
        if not input_path.exists():
            self.logger.error(f"Input file does not exist: {input_path.name}")
            return None
            
        # Check if the file is already optimized (for web processing)
        if "_optimized" in input_path.stem:
            self.logger.info(f"File appears to be already optimized: {input_path.name}")
            return input_path
            
        # Get file size in MB
        file_size_mb = input_path.stat().st_size / (1024 * 1024)
        
        # If downsize is not enabled or file is already smaller than max_size_mb, return the original file
        if not self.config.downsize or file_size_mb <= self.config.max_size_mb:
            self.logger.info(f"Skipping optimization: downsize={self.config.downsize}, file_size={file_size_mb:.2f}MB, max_size={self.config.max_size_mb}MB")
            return input_path
            
        # Only proceed with optimization if downsize is enabled and file is larger than max_size_mb
        self.logger.info(f"File size ({file_size_mb:.2f}MB) exceeds max size ({self.config.max_size_mb}MB). Downsizing...")
        
        try:
            # Create a safe output filename
            safe_filename = self._sanitize_filename(f"{input_path.stem}_optimized.{self.config.video_format}")
            output_path = temp_dir / safe_filename

            # Enhanced optimization with better video quality at smaller sizes
            with self.console.status(f"[bold blue]Downsizing video to {self.config.max_size_mb}MB...") as status:
                # Get video information for better targeting
                try:
                    probe = ffmpeg.probe(str(input_path))
                    # Find video stream
                    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
                    if not video_stream:
                        self.logger.error("No video stream found")
                        return input_path
                        
                    # Get video properties
                    duration = float(video_stream.get('duration', 0))
                    width = int(video_stream.get('width', 1280))
                    height = int(video_stream.get('height', 720))
                    fps = eval(video_stream.get('r_frame_rate', '30/1'))
                    
                    if duration <= 0:
                        self.logger.warning("Could not determine video duration, using default")
                        duration = 60  # Default 1 minute
                except Exception as probe_error:
                    self.logger.warning(f"Could not probe video: {probe_error}")
                    duration = 60  # Default 1 minute
                    width, height, fps = 1280, 720, 30
                
                # Calculate scaled dimensions to reduce resolution if needed
                # For very large videos, reducing resolution is much more efficient than just bitrate
                max_dimension = 720  # Limit to 720p for smaller files
                scale_factor = 1.0
                
                if height > max_dimension or width > max_dimension:
                    if height > width:
                        scale_factor = max_dimension / height
                    else:
                        scale_factor = max_dimension / width
                        
                new_width = int(width * scale_factor) - (int(width * scale_factor) % 2)  # Ensure even dimensions
                new_height = int(height * scale_factor) - (int(height * scale_factor) % 2)
                
                # Target size in bits (minus overhead)
                target_size_bits = self.config.max_size_mb * 8 * 1024 * 1024 * 0.9
                
                # Reserve audio bitrate (reduce for longer videos)
                audio_bitrate = min(128, int(2000 / max(1, duration/60)))  # Scale down for longer videos
                audio_bitrate_bits = audio_bitrate * 1024
                
                # Calculate video bitrate based on duration
                video_bitrate = int((target_size_bits / duration) - audio_bitrate_bits)
                video_bitrate = max(video_bitrate, 100 * 1024)  # Minimum 100kbps
                
                # For large compression ratios, use faster preset and higher CRF for better speed
                compression_ratio = file_size_mb / self.config.max_size_mb
                preset = 'veryfast'  # Use faster preset for large files
                
                # Determine max dimension based on compression ratio (smaller for larger compressions)
                if compression_ratio > 50:  # For extreme cases (e.g., 500MB to 8MB)
                    max_dimension = 480  # Use 480p for extreme compression
                    preset = 'ultrafast'  # Use fastest preset
                    crf = 40  # Maximum compression quality loss
                    # Additional reduction in FPS
                    fps_target = min(24, int(fps))  # Cap at 24fps or lower if original is lower
                    fps_filter = f',fps={fps_target}'  # Add FPS reduction filter
                elif compression_ratio > 20:
                    max_dimension = 540  # Use 540p
                    preset = 'ultrafast'
                    crf = 38
                    fps_filter = ''
                else:
                    max_dimension = 720  # Use 720p for smaller compression ratios
                    crf = min(35, 23 + int(compression_ratio / 10))
                    fps_filter = ''
                    
                # Recalculate dimensions based on max_dimension
                if height > width:
                    scale_factor = max_dimension / height
                else:
                    scale_factor = max_dimension / width
                
                self.logger.info(f"Compression ratio: {compression_ratio:.1f}x, preset: {preset}, CRF: {crf}")
                self.logger.info(f"Scaling from {width}x{height} to {new_width}x{new_height}")
                
                # Add keyframe optimization for longer videos
                keyframe_options = []
                if duration > 600:  # 10 minutes
                    keyframe_options = [
                        '-g', '250',  # Keyframe every 250 frames
                        '-sc_threshold', '0'  # Disable scene change detection
                    ]
                    self.logger.info("Using optimized keyframe settings for long video")
                
                # Add hardware acceleration if available
                hw_accel_options = []
                if self.hw_accel == 'cuda':
                    hw_accel_options = [
                        '-hwaccel', 'cuda',
                        '-c:v', 'h264_nvenc'
                    ]
                elif self.hw_accel == 'videotoolbox':
                    hw_accel_options = [
                        '-hwaccel', 'videotoolbox',
                        '-c:v', 'h264_videotoolbox'
                    ]
                elif self.hw_accel == 'qsv':
                    hw_accel_options = [
                        '-hwaccel', 'qsv',
                        '-c:v', 'h264_qsv'
                    ]
                elif self.hw_accel == 'vaapi':
                    hw_accel_options = [
                        '-hwaccel', 'vaapi',
                        '-vaapi_device', '/dev/dri/renderD128',
                        '-c:v', 'h264_vaapi'
                    ]
                
                # Security improvement: add -nostdin to prevent interactive prompts
                # Build FFmpeg command for fast optimization
                ffmpeg_command = [
                    'ffmpeg',
                    '-nostdin',  # Security improvement: disable stdin interaction
                    '-i', str(input_path)
                ]
                
                # Add hardware acceleration if available
                if hw_accel_options:
                    ffmpeg_command.extend(hw_accel_options)
                    
                # Continue with main command
                ffmpeg_command.extend([
                    '-vf', f'scale={new_width}:{new_height}{fps_filter}',
                    '-c:v', 'libx264' if not hw_accel_options else ffmpeg_command[-1],  # Keep hw encoder if set
                    '-preset', preset,
                    '-crf', str(crf),
                    '-b:v', f'{video_bitrate}',
                    '-maxrate', f'{video_bitrate * 1.2}',
                    '-bufsize', f'{video_bitrate}',
                    '-pix_fmt', 'yuv420p',  # Ensure compatibility
                    '-tune', 'film',  # Optimize for video content
                    '-c:a', 'aac',
                    '-b:a', f'{audio_bitrate}k',
                    '-ac', '2',  # Stereo audio
                    '-ar', '44100',  # Lower audio sample rate
                    '-movflags', '+faststart'  # Optimize for web streaming
                ])
                
                # Add keyframe options if needed
                if keyframe_options:
                    ffmpeg_command.extend(keyframe_options)
                    
                # Add remaining options
                ffmpeg_command.extend([
                    '-threads', '0',  # Use all available CPU cores
                    '-f', self.config.video_format,
                    '-y',  # Overwrite output
                    str(output_path)
                ])
                
                # Use a shorter timeout for better responsiveness
                timeout = min(600, max(60, int(duration * 0.8)))
                
                try:
                    # Start FFmpeg process with enhanced security
                    process = subprocess.Popen(
                        ffmpeg_command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        stdin=subprocess.DEVNULL,  # Explicitly prevent stdin
                        universal_newlines=True,
                        errors='replace'  # Handle potential encoding errors
                    )
                    
                    # Start a thread to monitor progress
                    with Status("[bold blue]Downsizing video... analyzing", console=self.console) as status:
                        # Start a thread to monitor progress
                        progress_thread = threading.Thread(
                            target=self._monitor_ffmpeg_progress,
                            args=(process, duration, status),
                            daemon=True
                        )
                        progress_thread.start()
                        
                        # Wait for the process to complete with timeout
                        stdout, stderr = process.communicate(timeout=timeout)
                    
                    # Check return code
                    if process.returncode != 0:
                        self.logger.error(f"FFmpeg downsizing failed: {stderr}")
                        return input_path  # Return original if optimization fails
                        
                except subprocess.TimeoutExpired:
                    process.kill()
                    self.logger.error("Downsizing timed out")
                    return input_path  # Return original if timeout
                    
                except Exception as e:
                    self.logger.error(f"Downsizing error: {str(e)}")
                    return input_path  # Return original on error
            
            # Verify output file was created and is smaller
            if not output_path.exists() or output_path.stat().st_size == 0:
                self.logger.error("Downsizing failed: Output file is empty or not created")
                return input_path  # Return original if output is invalid
                
            # Check if the output is actually smaller
            output_size_mb = output_path.stat().st_size / (1024 * 1024)
            if output_size_mb >= file_size_mb:
                self.logger.warning(f"Downsized file ({output_size_mb:.2f}MB) is not smaller than original ({file_size_mb:.2f}MB)")
                return input_path  # Return original if no size reduction
                
            self.logger.info(f"Video downsized: {output_path.name} ({output_size_mb:.2f}MB)")
            
            # Add successfully downsized video to cache
            video_id = self._extract_video_id(self.config.url)
            if video_id:
                self._add_to_cache(video_id, self.config.quality, self.config.video_format, output_path)
                
            return output_path

        except Exception as e:
            self.logger.error(f"Downsizing error: {str(e)}")
            return input_path  # Return original on any error

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
        """Process video from URL to optimized video and audio files using concurrent processing"""
        result = {
            "success": False,
            "video_path": None,
            "audio_path": None,
            "error": None,
            "output_dir": str(self.output_dir),
            "run_id": self.run_uuid,
            "timestamp": self._get_current_timestamp()
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
                # Use threading for parallel processing of video and audio
                processing_results = {}
                processing_threads = []

                # Setup video processing thread if needed
                if not self.config.audio_only:
                    file_size_mb = downloaded_video.stat().st_size / (1024 * 1024)
                    self.logger.info(f"Downloaded video size: {file_size_mb:.2f}MB")
                    
                    # Only optimize if file is bigger than max_size and downsizing is enabled
                    if self.config.downsize and file_size_mb > self.config.max_size_mb:
                        video_thread = threading.Thread(
                            target=self._process_video_thread,
                            args=(downloaded_video, temp_dir, processing_results)
                        )
                    else:
                        # Just copy the downloaded video without optimization
                        video_thread = threading.Thread(
                            target=self._copy_video_thread,
                            args=(downloaded_video, processing_results)
                        )
                    
                    processing_threads.append(video_thread)
                    video_thread.start()

                # Setup audio processing thread if needed
                if not self.config.video_only:
                    audio_thread = threading.Thread(
                        target=self._process_audio_thread,
                        args=(downloaded_video, temp_dir, processing_results)
                    )
                    processing_threads.append(audio_thread)
                    audio_thread.start()

                # Wait for all threads to complete
                for thread in processing_threads:
                    thread.join()

                # Update result with thread processing outputs
                if "video_path" in processing_results:
                    result["video_path"] = processing_results["video_path"]
                
                if "audio_path" in processing_results:
                    result["audio_path"] = processing_results["audio_path"]

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
                        "output_directory": str(self.output_dir),
                        "run_id": self.run_uuid,
                        "video_filename": result["video_path"].name if result["video_path"] else None,
                        "audio_filename": result["audio_path"].name if result["audio_path"] else None,
                        "url": str(self.config.url)[:20] + "..."  # Redacted for privacy
                    }
                    self._log_details(log_details)
                    
                    # Save metadata for web access
                    self._save_metadata(log_details)

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
            self.logger.error(f"Failed to write log file: {str(e)}")
            
    def _save_metadata(self, details: Dict[str, Any]):
        """Save metadata for web access"""
        try:
            # Create a web-friendly metadata file
            web_metadata = {
                "id": self.run_uuid,
                "timestamp": details["timestamp"],
                "files": [],
                "status": "completed",
                "url": details.get("url", "URL redacted"),
                "quality": details.get("quality_setting", self.config.quality)
            }
            
            # Add video file if present
            if details.get("video_processed") and details.get("video_filename"):
                web_metadata["files"].append({
                    "type": "video",
                    "filename": details["video_filename"],
                    "format": self.config.video_format,
                    "hash": details.get("video_hash", "")
                })
                
            # Add audio file if present
            if details.get("audio_processed") and details.get("audio_filename"):
                web_metadata["files"].append({
                    "type": "audio",
                    "filename": details["audio_filename"],
                    "format": self.config.audio_format,
                    "hash": details.get("audio_hash", "")
                })
                
            # Write the metadata file
            with open(self.metadata_path, "w") as f:
                json.dump(web_metadata, f, indent=4)
                
            self.logger.info(f"Web metadata saved to {self.metadata_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save web metadata: {str(e)}")
            
    def _process_video_thread(self, input_video: Path, temp_dir: Path, results: Dict[str, Any]):
        """Thread function to process video with optional optimization"""
        try:
            # Check if we need to optimize (reduce file size)
            file_size_mb = input_video.stat().st_size / (1024 * 1024)
            if self.config.downsize and file_size_mb > self.config.max_size_mb:
                self.logger.info(f"Video file size: {file_size_mb:.2f}MB, max size: {self.config.max_size_mb}MB")
                optimized_video = self.optimize_video(input_video, temp_dir)
                if optimized_video:
                    video_path = self.copy_to_output_dir(optimized_video)
                    if video_path:
                        results["video_path"] = video_path
                        # Verify integrity
                        original_hash = self.hash_file(optimized_video)
                        if not self.verify_hash(video_path, original_hash):
                            self.logger.warning("Video hash verification failed - possible corruption")
            else:
                # Just copy the original file
                self.logger.info(f"Skipping optimization, copying original video")
                video_path = self.copy_to_output_dir(input_video)
                if video_path:
                    results["video_path"] = video_path
        except Exception as e:
            self.logger.error(f"Error in video processing thread: {str(e)}")
            
    def _copy_video_thread(self, input_video: Path, results: Dict[str, Any]):
        """Thread function to just copy video without optimization"""
        try:
            video_path = self.copy_to_output_dir(input_video)
            if video_path:
                results["video_path"] = video_path
                # Verify integrity
                original_hash = self.hash_file(input_video)
                if not self.verify_hash(video_path, original_hash):
                    self.logger.warning("Video hash verification failed - possible corruption")
        except Exception as e:
            self.logger.error(f"Error in video copy thread: {str(e)}")
    
    def _process_audio_thread(self, input_video: Path, temp_dir: Path, results: Dict[str, Any]):
        """Thread function to extract audio"""
        try:
            audio_file = self.extract_audio(input_video, temp_dir)
            if audio_file:
                audio_path = self.copy_to_output_dir(audio_file)
                if audio_path:
                    results["audio_path"] = audio_path
                    # Verify integrity
                    original_hash = self.hash_file(audio_file)
                    if not self.verify_hash(audio_path, original_hash):
                        self.logger.warning("Audio hash verification failed - possible corruption")
        except Exception as e:
            self.logger.error(f"Error in audio processing thread: {str(e)}")
