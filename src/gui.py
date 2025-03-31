#!/usr/bin/env python3
"""
YouTube Video Downloader & Optimizer - GUI
A graphical interface for the YouTube downloader application.
"""

import os
import sys
import threading
import logging
import tempfile
import time
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

import dearpygui.dearpygui as dpg
from rich.console import Console

from models import VideoDownloadConfig
from downloader import YouTubeDownloader
from batch import BatchProcessor
from cli import run_batch_mode, run_playlist_mode


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("yt_downloader_gui")


class YouTubeDownloaderGUI:
    """Main GUI class for the YouTube downloader application"""
    
    def __init__(self):
        """Initialize the GUI application"""
        self.console = Console(record=True)
        self.log_level = logging.INFO
        
        # Application state
        self.is_downloading = False
        self.download_thread = None
        self.current_output_dir = str(Path.cwd())
        self.log_messages = []
        self.download_progress = 0.0
        self.download_status = "Ready"
        
        # Configuration defaults
        self.url = ""
        self.input_file = ""
        self.quality = "high"
        self.format = "mp4"
        self.audio_format = "mp3"
        self.audio_quality = "320k"
        self.video_only = False
        self.audio_only = False
        self.downsize = False
        self.max_size_mb = 8
        self.workers = 2
        
    def setup_gui(self):
        """Set up the Dear PyGui interface"""
        dpg.create_context()
        
        # Create viewport
        dpg.create_viewport(
            title="YouTube Downloader & Optimizer",
            width=900,
            height=650,
            resizable=True
        )
        
        # Setup theme
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (37, 37, 38, 255))
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (15, 15, 15, 255))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (40, 40, 40, 255))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (40, 40, 40, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (59, 59, 59, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (66, 150, 250, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (66, 150, 250, 255))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (66, 150, 250, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255))
        
        dpg.bind_theme(global_theme)
        
        # Use the default font
        # The custom font loading was causing issues because it requires a path to a font file
        
        # Create the main window
        with dpg.window(tag="main_window", label="YouTube Downloader", width=900, height=650):
            self._create_tabs()
        
        # Setup callbacks for DPG
        dpg.setup_dearpygui()
        dpg.show_viewport()
        
    def _create_tabs(self):
        """Create the tab bar with different modes"""
        with dpg.tab_bar():
            with dpg.tab(label="Single Download"):
                self._create_single_download_tab()
            
            with dpg.tab(label="Batch Processing"):
                self._create_batch_tab()
            
            with dpg.tab(label="Playlist Processing"):
                self._create_playlist_tab()
            
            with dpg.tab(label="Settings"):
                self._create_settings_tab()

    def _create_single_download_tab(self):
        """Create the UI for single video download mode"""
        dpg.add_text("Download a single YouTube video")
        dpg.add_separator()
        
        # URL input
        dpg.add_input_text(
            label="YouTube URL",
            width=-1,
            callback=self._on_url_change,
            hint="https://www.youtube.com/watch?v=...",
            tag="single_url"
        )
        
        # Output directory
        with dpg.group(horizontal=True):
            dpg.add_input_text(
                label="Output Directory",
                default_value=self.current_output_dir,
                width=-75,
                tag="single_output_dir"
            )
            dpg.add_button(
                label="Browse",
                callback=lambda: self._browse_directory("single_output_dir")
            )
        
        # Common options
        self._add_common_options("single_")
        
        # Download button
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Download",
                callback=self._start_single_download,
                width=200
            )
            dpg.add_button(
                label="Cancel",
                callback=self._cancel_download,
                width=150
            )
        
        # Progress bar and status
        dpg.add_separator()
        dpg.add_text("Status: Ready", tag="status_text")
        dpg.add_progress_bar(default_value=0.0, width=-1, tag="progress_bar")
        
        # Log output
        dpg.add_text("Log Output")
        dpg.add_input_text(
            multiline=True,
            readonly=True,
            width=-1,
            height=250,
            tag="log_output"
        )
    
    def _create_batch_tab(self):
        """Create the UI for batch processing mode"""
        dpg.add_text("Batch process multiple YouTube videos")
        dpg.add_separator()
        
        # Input file with URLs
        with dpg.group(horizontal=True):
            dpg.add_input_text(
                label="Input File (URLs)",
                width=-75,
                tag="batch_input_file",
                hint="Path to file with one URL per line"
            )
            dpg.add_button(
                label="Browse",
                callback=self._browse_input_file
            )
        
        # Output directory
        with dpg.group(horizontal=True):
            dpg.add_input_text(
                label="Output Directory",
                default_value=self.current_output_dir,
                width=-75,
                tag="batch_output_dir"
            )
            dpg.add_button(
                label="Browse",
                callback=lambda: self._browse_directory("batch_output_dir")
            )
        
        # Workers
        dpg.add_slider_int(
            label="Worker Threads",
            default_value=2,
            min_value=1,
            max_value=8,
            width=300,
            tag="batch_workers"
        )
        
        # Common options
        self._add_common_options("batch_")
        
        # Start batch button
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Start Batch Processing",
                callback=self._start_batch_processing,
                width=200
            )
            dpg.add_button(
                label="Cancel",
                callback=self._cancel_download,
                width=150
            )
        
        # Progress and status
        dpg.add_separator()
        dpg.add_text("Status: Ready", tag="batch_status_text")
        dpg.add_progress_bar(default_value=0.0, width=-1, tag="batch_progress_bar")
        
        # Batch processing stats
        with dpg.table(header_row=True, resizable=True, width=-1, height=250):
            dpg.add_table_column(label="Job ID")
            dpg.add_table_column(label="URL")
            dpg.add_table_column(label="Status")
            dpg.add_table_column(label="Time")
            dpg.add_table_column(label="Output")
            
            # Table will be populated dynamically
            dpg.add_table_row(tag="batch_table")
    
    def _create_playlist_tab(self):
        """Create the UI for playlist processing mode"""
        dpg.add_text("Process all videos from a YouTube playlist")
        dpg.add_separator()
        
        # Playlist URL
        dpg.add_input_text(
            label="YouTube Playlist URL",
            width=-1,
            tag="playlist_url",
            hint="https://www.youtube.com/playlist?list=..."
        )
        
        # Output directory
        with dpg.group(horizontal=True):
            dpg.add_input_text(
                label="Output Directory",
                default_value=self.current_output_dir,
                width=-75,
                tag="playlist_output_dir"
            )
            dpg.add_button(
                label="Browse",
                callback=lambda: self._browse_directory("playlist_output_dir")
            )
        
        # Workers
        dpg.add_slider_int(
            label="Worker Threads",
            default_value=2,
            min_value=1,
            max_value=8,
            width=300,
            tag="playlist_workers"
        )
        
        # Common options
        self._add_common_options("playlist_")
        
        # Start playlist button
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Process Playlist",
                callback=self._start_playlist_processing,
                width=200
            )
            dpg.add_button(
                label="Cancel",
                callback=self._cancel_download,
                width=150
            )
        
        # Progress and status
        dpg.add_separator()
        dpg.add_text("Status: Ready", tag="playlist_status_text")
        dpg.add_progress_bar(default_value=0.0, width=-1, tag="playlist_progress_bar")
        
        # Playlist processing stats
        with dpg.table(header_row=True, resizable=True, width=-1, height=250):
            dpg.add_table_column(label="Video #")
            dpg.add_table_column(label="Title")
            dpg.add_table_column(label="Status")
            dpg.add_table_column(label="Time")
            dpg.add_table_column(label="Output")
            
            # Table will be populated dynamically
            dpg.add_table_row(tag="playlist_table")
            
    def _create_settings_tab(self):
        """Create the UI for application settings"""
        dpg.add_text("Application Settings")
        dpg.add_separator()
        
        # Default output directory
        with dpg.group(horizontal=True):
            dpg.add_input_text(
                label="Default Output Directory",
                default_value=self.current_output_dir,
                width=-75,
                tag="default_output_dir",
                callback=self._update_default_output_dir
            )
            dpg.add_button(
                label="Browse",
                callback=lambda: self._browse_directory("default_output_dir")
            )
        
        # Default workers
        dpg.add_slider_int(
            label="Default Worker Threads",
            default_value=self.workers,
            min_value=1,
            max_value=8,
            width=300,
            tag="default_workers",
            callback=self._update_default_workers
        )
        
        # Default quality
        dpg.add_combo(
            label="Default Quality",
            items=["high", "medium", "low"],
            default_value=self.quality,
            width=200,
            tag="default_quality",
            callback=self._update_default_quality
        )
        
        # Default format
        dpg.add_input_text(
            label="Default Video Format",
            default_value=self.format,
            width=200,
            tag="default_format",
            callback=self._update_default_format
        )
        
        # Default audio format
        dpg.add_input_text(
            label="Default Audio Format",
            default_value=self.audio_format,
            width=200,
            tag="default_audio_format",
            callback=self._update_default_audio_format
        )
        
        # Default audio quality
        dpg.add_input_text(
            label="Default Audio Quality",
            default_value=self.audio_quality,
            width=200,
            tag="default_audio_quality",
            callback=self._update_default_audio_quality
        )
        
        # Max size settings
        dpg.add_checkbox(
            label="Enable Video Downsizing",
            default_value=self.downsize,
            tag="default_downsize",
            callback=self._update_default_downsize
        )
        
        dpg.add_slider_int(
            label="Max Size (MB)",
            default_value=self.max_size_mb,
            min_value=1,
            max_value=50,
            width=300,
            tag="default_max_size_mb",
            callback=self._update_default_max_size
        )
        
        # About section
        dpg.add_separator()
        dpg.add_text("About")
        dpg.add_text("YouTube Downloader & Optimizer")
        dpg.add_text("A GUI for downloading and processing YouTube videos")
        
    def _add_common_options(self, prefix=""):
        """Add common download options that appear in multiple tabs"""
        # Format selection
        with dpg.collapsing_header(label="Format Options", default_open=True):
            # Quality selection
            dpg.add_combo(
                label="Quality",
                items=["high", "medium", "low"],
                default_value=self.quality,
                width=200,
                tag=f"{prefix}quality"
            )
            
            # Format selection
            dpg.add_input_text(
                label="Video Format",
                default_value=self.format,
                width=200,
                tag=f"{prefix}format"
            )
            
            # Media type selection
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="Video Only",
                    default_value=self.video_only,
                    tag=f"{prefix}video_only",
                    callback=lambda sender, data: self._handle_media_checkbox(f"{prefix}video_only", f"{prefix}audio_only", data)
                )
                dpg.add_checkbox(
                    label="Audio Only",
                    default_value=self.audio_only,
                    tag=f"{prefix}audio_only",
                    callback=lambda sender, data: self._handle_media_checkbox(f"{prefix}audio_only", f"{prefix}video_only", data)
                )
        
        # Advanced options
        with dpg.collapsing_header(label="Advanced Options"):
            # Audio format and quality
            dpg.add_input_text(
                label="Audio Format",
                default_value=self.audio_format,
                width=200,
                tag=f"{prefix}audio_format"
            )
            
            dpg.add_input_text(
                label="Audio Quality",
                default_value=self.audio_quality,
                width=200,
                tag=f"{prefix}audio_quality"
            )
            
            # Downsizing
            dpg.add_checkbox(
                label="Enable Video Downsizing",
                default_value=self.downsize,
                tag=f"{prefix}downsize"
            )
            
            dpg.add_slider_int(
                label="Max Size (MB)",
                default_value=self.max_size_mb,
                min_value=1,
                max_value=50,
                width=300,
                tag=f"{prefix}max_size_mb"
            )
    
    # Callback methods
    def _handle_media_checkbox(self, checkbox_id, other_checkbox_id, value):
        """Handle the exclusive selection between audio-only and video-only"""
        if value and dpg.get_value(other_checkbox_id):
            dpg.set_value(other_checkbox_id, False)
    
    def _on_url_change(self, sender, data):
        """Handle URL input changes with enhanced validation"""
        self.url = data
        
        # Validate URL format if we have data
        if data and not self._is_valid_youtube_url(data):
            self._update_status("Warning: URL doesn't appear to be a valid YouTube URL")
        else:
            self._update_status("")
            
    def _is_valid_youtube_url(self, url: str, playlist: bool = False) -> bool:
        """Validate if a string is a properly formatted YouTube URL
        
        Args:
            url: The URL to validate
            playlist: Whether to validate as a playlist URL
            
        Returns:
            bool: True if the URL is valid, False otherwise
        """
        if not url or not isinstance(url, str):
            return False
            
        # Security: Sanitize URL - remove potentially dangerous characters
        if any(char in url for char in [';', '&', '|', '`', '$', '\\']):
            return False
            
        # Basic YouTube URL patterns
        youtube_domains = [
            r'(https?://)?(www\.)?(youtube\.com|youtu\.be)',
            r'(https?://)?(www\.)?(youtube-nocookie\.com)'
        ]
        
        domain_pattern = '|'.join(youtube_domains)
        
        if playlist:
            # Validate playlist URL format
            playlist_patterns = [
                # Standard playlist URL
                fr'{domain_pattern}/playlist\?list=[\w-]+',
                # Watch URL with playlist parameter
                fr'{domain_pattern}/watch\?v=[\w-]+&list=[\w-]+',
                fr'{domain_pattern}/watch\?list=[\w-]+&v=[\w-]+'
            ]
            pattern = '|'.join(playlist_patterns)
        else:
            # Validate single video URL format
            video_patterns = [
                # Standard watch URL
                fr'{domain_pattern}/watch\?v=[\w-]+',
                # Short URL
                fr'{domain_pattern}/shorts/[\w-]+',
                # Embed URL
                fr'{domain_pattern}/embed/[\w-]+',
                # Short URL format
                fr'youtu\.be/[\w-]+'
            ]
            pattern = '|'.join(video_patterns)
        
        return bool(re.match(pattern, url))
    
    def _update_default_output_dir(self, sender, data):
        """Update the default output directory"""
        self.current_output_dir = data
        # Update all output directory fields
        dpg.set_value("single_output_dir", data)
        dpg.set_value("batch_output_dir", data)
        dpg.set_value("playlist_output_dir", data)
    
    def _update_default_workers(self, sender, data):
        """Update the default worker count"""
        self.workers = data
        dpg.set_value("batch_workers", data)
        dpg.set_value("playlist_workers", data)
    
    def _update_default_quality(self, sender, data):
        """Update the default quality"""
        self.quality = data
    
    def _update_default_format(self, sender, data):
        """Update the default format"""
        self.format = data
    
    def _update_default_audio_format(self, sender, data):
        """Update the default audio format"""
        self.audio_format = data
    
    def _update_default_audio_quality(self, sender, data):
        """Update the default audio quality"""
        self.audio_quality = data
    
    def _update_default_downsize(self, sender, data):
        """Update the default downsize setting"""
        self.downsize = data
    
    def _update_default_max_size(self, sender, data):
        """Update the default max size"""
        self.max_size_mb = data
    
    def _browse_directory(self, tag_id):
        """Open a file dialog to browse for a directory with enhanced security"""
        def callback(sender, app_data):
            # Security: Validate directory path before accepting
            try:
                path = app_data['file_path_name']
                path_obj = Path(path)
                
                # Basic path validation
                if not path_obj.is_dir():
                    self._update_status(f"Selected path is not a directory: {path}")
                    return
                    
                # Check write permissions
                if not os.access(path_obj, os.W_OK):
                    self._update_status(f"No write permission to selected directory: {path}")
                    return
                    
                # Path is valid and writable, update UI
                dpg.set_value(tag_id, path)
                if tag_id == "default_output_dir":
                    self._update_default_output_dir(None, path)
                    
                self._update_status(f"Directory set: {path}")
            except Exception as e:
                self._update_status(f"Error validating directory: {str(e)}")
        
        with dpg.file_dialog(directory_selector=True, show=True, callback=callback):
            dpg.add_file_extension(".*")
    
    def _browse_input_file(self):
        """Open a file dialog to browse for an input file"""
        def callback(sender, app_data):
            dpg.set_value("batch_input_file", app_data['file_path_name'])
            self.input_file = app_data['file_path_name']
        
        with dpg.file_dialog(directory_selector=False, show=True, callback=callback):
            dpg.add_file_extension(".txt", color=(0, 255, 0, 255))
            dpg.add_file_extension(".*")
    
    def _get_config_from_ui(self, prefix="single_"):
        """Get the download configuration from UI elements with enhanced security validation"""
        # Get and validate output directory
        output_dir = dpg.get_value(f"{prefix}output_dir")
        if not output_dir:
            output_dir = self.current_output_dir
        
        # Security: Validate output directory
        try:
            output_path = Path(output_dir)
            # Ensure path exists or can be created
            if not output_path.exists():
                try:
                    output_path.mkdir(parents=True, exist_ok=True)
                except (PermissionError, OSError) as e:
                    self._update_status(f"Error creating output directory: {str(e)}", prefix=prefix.rstrip('_'))
                    return None
            # Ensure we have write permissions
            elif not os.access(output_path, os.W_OK):
                self._update_status(f"No write permission to output directory", prefix=prefix.rstrip('_'))
                return None
        except Exception as e:
            self._update_status(f"Invalid output directory path: {str(e)}", prefix=prefix.rstrip('_'))
            return None
        
        # Get and validate URLs
        if prefix == "single_":
            url = dpg.get_value("single_url")
            # Security: Enhanced URL validation
            if not url or not self._is_valid_youtube_url(url):
                self._update_status("Please enter a valid YouTube URL")
                return None
        elif prefix == "playlist_":
            url = dpg.get_value("playlist_url")
            # Security: Enhanced playlist URL validation
            if not url or not self._is_valid_youtube_url(url, playlist=True):
                self._update_status("Please enter a valid YouTube playlist URL", prefix="playlist_")
                return None
        
        # Create config
        try:
            config = {
                "url": url if prefix != "batch_" else None,
                "output_dir": Path(output_dir),
                "quality": dpg.get_value(f"{prefix}quality"),
                "format": dpg.get_value(f"{prefix}format"),
                "video_only": dpg.get_value(f"{prefix}video_only"),
                "audio_only": dpg.get_value(f"{prefix}audio_only"),
                "audio_format": dpg.get_value(f"{prefix}audio_format"),
                "audio_quality": dpg.get_value(f"{prefix}audio_quality"),
                "downsize": dpg.get_value(f"{prefix}downsize"),
                "max_size_mb": dpg.get_value(f"{prefix}max_size_mb"),
            }
            
            # Special handling for batch/playlist
            if prefix == "batch_":
                config["input_file"] = dpg.get_value("batch_input_file")
                config["workers"] = dpg.get_value("batch_workers")
            elif prefix == "playlist_":
                config["workers"] = dpg.get_value("playlist_workers")
            
            return config
            
        except Exception as e:
            self._update_status(f"Error creating configuration: {str(e)}")
            logger.error(f"Error creating configuration: {str(e)}")
            return None
    
    def _update_status(self, message, prefix=""):
        """Update the status message"""
        status_tag = f"{prefix}status_text" if prefix else "status_text"
        if dpg.does_item_exist(status_tag):
            dpg.set_value(status_tag, f"Status: {message}")
        self._add_log(message)
    
    def _update_progress(self, progress, prefix=""):
        """Update the progress bar"""
        progress_tag = f"{prefix}progress_bar" if prefix else "progress_bar"
        if dpg.does_item_exist(progress_tag):
            dpg.set_value(progress_tag, progress)
    
    def _add_log(self, message):
        """Add a message to the log output"""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self.log_messages.append(log_message)
        
        # Update log display
        if dpg.does_item_exist("log_output"):
            # Keep only the last 100 log messages to avoid performance issues
            if len(self.log_messages) > 100:
                self.log_messages = self.log_messages[-100:]
            dpg.set_value("log_output", "\n".join(self.log_messages))
    
    def _start_single_download(self):
        """Start downloading a single video"""
        if self.is_downloading:
            self._update_status("Download already in progress")
            return
        
        config = self._get_config_from_ui("single_")
        if not config:
            return
        
        self.is_downloading = True
        self._update_status("Starting download...")
        self._update_progress(0.0)
        
        # Start download in a separate thread
        self.download_thread = threading.Thread(
            target=self._download_single_video,
            args=(config,),
            daemon=True
        )
        self.download_thread.start()
    
    def _download_single_video(self, config):
        """Download a single video in a separate thread"""
        temp_dir = None
        try:
            # Create video download config
            download_config = VideoDownloadConfig(
                url=config["url"],
                output_dir=config["output_dir"],
                quality=config["quality"],
                format=config["format"],
                video_only=config["video_only"],
                audio_only=config["audio_only"],
                audio_format=config["audio_format"],
                audio_quality=config["audio_quality"],
                downsize=config["downsize"],
                max_size_mb=config["max_size_mb"]
            )
            
            # Create downloader
            downloader = YouTubeDownloader(download_config)
            
            # Set up progress callback
            def progress_callback(progress, status):
                # Ensure UI updates happen in the main thread
                dpg.configure_item("progress_bar", default_value=progress)
                dpg.set_value("status_text", status)
            
            downloader.set_progress_callback(progress_callback)
            
            # Create a temporary directory for the download
            import tempfile
            temp_dir = Path(tempfile.mkdtemp(prefix="yt_downloader_"))
            
            # Download the video
            self._update_status("Preparing download...")
            self._update_progress(0.0)
            
            # Call download_video with the temp directory
            result = downloader.download_video(temp_dir)
            
            if result:
                # Copy the file to the output directory if not already there
                output_path = Path(config["output_dir"]) / result.name
                if result.parent != output_path.parent:
                    import shutil
                    shutil.copy2(result, output_path)
                    final_result = output_path
                else:
                    final_result = result
                    
                self._update_status(f"Download complete: {final_result.name}")
                self._update_progress(1.0)
            else:
                self._update_status("Download failed")
                self._update_progress(0.0)
                
        except Exception as e:
            self._update_status(f"Error: {str(e)}")
            logger.error(f"Download error: {str(e)}")
            self._update_progress(0.0)
        finally:
            # Clean up the temporary directory
            if temp_dir and temp_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.error(f"Failed to clean up temp directory: {str(e)}")
            self.is_downloading = False
    
    def _start_batch_processing(self):
        """Start batch processing of videos"""
        if self.is_downloading:
            self._update_status("Batch processing already in progress", prefix="batch_")
            return
        
        config = self._get_config_from_ui("batch_")
        if not config or not config["input_file"]:
            self._update_status("Please select an input file", prefix="batch_")
            return
        
        self.is_downloading = True
        self._update_status("Starting batch processing...", prefix="batch_")
        self._update_progress(0.0, prefix="batch_")
        
        # Start batch processing in a separate thread
        self.download_thread = threading.Thread(
            target=self._process_batch,
            args=(config,),
            daemon=True
        )
        self.download_thread.start()
    
    def _process_batch(self, config):
        """Process a batch of videos in a separate thread"""
        try:
            # Parse input file - fix any path issues
            input_file = config["input_file"]
            
            # Fix common path issues (like extra characters at the end)
            if not Path(input_file).exists() and input_file.endswith('ba'):
                # Try removing the 'ba' at the end if that's the issue
                corrected_path = input_file[:-2]
                if Path(corrected_path).exists():
                    input_file = corrected_path
                    self._update_status(f"Corrected file path to: {input_file}", prefix="batch_")
            
            # Check if file exists
            if not Path(input_file).exists():
                self._update_status(f"Input file not found: {input_file}", prefix="batch_")
                self.is_downloading = False
                return
                
            try:
                with open(input_file, "r") as f:
                    urls = [line.strip() for line in f if line.strip()]
            except Exception as e:
                self._update_status(f"Error reading input file: {str(e)}", prefix="batch_")
                logger.error(f"Error reading batch file: {str(e)}")
                self.is_downloading = False
                return
            
            if not urls:
                self._update_status("Input file contains no valid URLs", prefix="batch_")
                self.is_downloading = False
                return
            
            self._update_status(f"Found {len(urls)} URLs to process", prefix="batch_")
            
            # Create base config for each download
            base_config = {
                "output_dir": config["output_dir"],
                "quality": config["quality"],
                "format": config["format"],
                "video_only": config["video_only"],
                "audio_only": config["audio_only"],
                "audio_format": config["audio_format"],
                "audio_quality": config["audio_quality"],
                "downsize": config["downsize"],
                "max_size_mb": config["max_size_mb"]
            }
            
            # Initialize batch processor with the correct parameters
            base_output_dir = Path(config["output_dir"])
            processor = BatchProcessor(
                base_output_dir=base_output_dir,
                max_workers=config["workers"],
                log_level=self.log_level
            )
            
            # Add each URL as a job to the batch processor
            for url in urls:
                # Create a VideoDownloadConfig for this URL
                download_config = VideoDownloadConfig(
                    url=url,
                    output_dir=base_output_dir,
                    quality=config["quality"],
                    format=config["format"],
                    video_only=config["video_only"],
                    audio_only=config["audio_only"],
                    audio_format=config["audio_format"],
                    audio_quality=config["audio_quality"],
                    downsize=config["downsize"],
                    max_size_mb=config["max_size_mb"]
                )
                
                # Add this job to the processor
                processor.add_job(download_config)
            
            # Track batch progress
            total_jobs = len(urls)
            completed_jobs = 0
            
            # Create a thread to monitor batch progress
            def monitor_batch_progress():
                nonlocal completed_jobs
                while processor.processing and self.is_downloading:
                    # Get current job statuses
                    job_statuses = processor.list_jobs()
                    
                    # Count completed jobs
                    new_completed = sum(1 for job in job_statuses if job["status"] in ["completed", "failed"])
                    
                    if new_completed != completed_jobs:
                        completed_jobs = new_completed
                        
                    # Update progress bar
                    progress = completed_jobs / total_jobs if total_jobs > 0 else 0
                    status_message = f"Processing {completed_jobs}/{total_jobs} videos"
                    
                    # Update UI
                    self._update_progress(progress, prefix="batch_")
                    self._update_status(status_message, prefix="batch_")
                    
                    # Log progress
                    if completed_jobs > 0 and completed_jobs % 5 == 0:
                        self._add_log(f"Batch progress: {completed_jobs}/{total_jobs} videos processed")
                    
                    time.sleep(1)
            
            # Start the monitoring thread
            monitor_thread = threading.Thread(target=monitor_batch_progress, daemon=True)
            monitor_thread.start()
            
            # Start batch processing
            self._update_status("Processing batch...", prefix="batch_")
            processor.start_processing()
            
            # Wait for processing to complete
            while processor.processing and self.is_downloading:
                time.sleep(0.5)  # Check more frequently but let the monitor thread handle updates
            
            # Display results
            processor.display_results()
            
            self._update_status("Batch processing complete", prefix="batch_")
            self._update_progress(1.0, prefix="batch_")
            
        except Exception as e:
            self._update_status(f"Error: {str(e)}", prefix="batch_")
            logger.error(f"Batch processing error: {str(e)}")
        finally:
            self.is_downloading = False
    
    def _start_playlist_processing(self):
        """Start processing a YouTube playlist"""
        if self.is_downloading:
            self._update_status("Playlist processing already in progress", prefix="playlist_")
            return
        
        config = self._get_config_from_ui("playlist_")
        if not config:
            return
        
        self.is_downloading = True
        self._update_status("Starting playlist processing...", prefix="playlist_")
        self._update_progress(0.0, prefix="playlist_")
        
        # Start playlist processing in a separate thread
        self.download_thread = threading.Thread(
            target=self._process_playlist,
            args=(config,),
            daemon=True
        )
        self.download_thread.start()
    
    def _process_playlist(self, config):
        """Process a YouTube playlist in a separate thread"""
        try:
            # Extract playlist videos
            self._update_status("Extracting playlist videos...", prefix="playlist_")
            self._update_progress(0.1, prefix="playlist_")
            
            # Create video download config for the playlist URL
            playlist_config = VideoDownloadConfig(
                url=config["url"],
                output_dir=config["output_dir"],
                quality=config["quality"],
                format=config["format"],
                audio_only=config["audio_only"],
                audio_format=config["audio_format"],
                audio_quality=config["audio_quality"]
            )
            
            # Create a temporary downloader just to extract playlist information
            import tempfile
            temp_dir = Path(tempfile.mkdtemp(prefix="yt_playlist_"))
            
            try:
                # Use yt-dlp directly to extract playlist info
                import yt_dlp
                
                ydl_opts = {
                    'quiet': True,
                    'extract_flat': True,  # Don't download videos
                    'dump_single_json': True,  # Return JSON data
                    'skip_download': True,  # Don't download videos
                    'outtmpl': str(temp_dir / "%(title)s.%(ext)s")
                }
                
                self._update_status("Retrieving playlist information...", prefix="playlist_")
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    playlist_info = ydl.extract_info(str(config["url"]), download=False)
                    
                    if not playlist_info or 'entries' not in playlist_info:
                        raise Exception("Could not find playlist entries")
                    
                    video_urls = [entry['url'] for entry in playlist_info['entries'] if 'url' in entry]
                    
                    playlist_title = playlist_info.get('title', 'Unknown Playlist')
                    self._update_status(f"Found {len(video_urls)} videos in playlist: {playlist_title}", prefix="playlist_")
                    self._update_progress(0.2, prefix="playlist_")
                    
                    if not video_urls:
                        raise Exception("No videos found in playlist")
                    
                    # Create base config for each download
                    base_config = {
                        "output_dir": config["output_dir"],
                        "quality": config["quality"],
                        "format": config["format"],
                        "video_only": config["video_only"],
                        "audio_only": config["audio_only"],
                        "audio_format": config["audio_format"],
                        "audio_quality": config["audio_quality"],
                        "downsize": config["downsize"],
                        "max_size_mb": config["max_size_mb"]
                    }
                    
                    # Initialize batch processor with the correct parameters
                    from batch import BatchProcessor
                    base_output_dir = Path(config["output_dir"])
                    processor = BatchProcessor(
                        base_output_dir=base_output_dir,
                        max_workers=config["workers"],
                        log_level=self.log_level
                    )
                    
                    # Add each URL as a job to the batch processor
                    for url in video_urls:
                        # Create a VideoDownloadConfig for this URL
                        download_config = VideoDownloadConfig(
                            url=url,
                            output_dir=base_output_dir,
                            quality=config["quality"],
                            format=config["format"],
                            video_only=config["video_only"],
                            audio_only=config["audio_only"],
                            audio_format=config["audio_format"],
                            audio_quality=config["audio_quality"],
                            downsize=config["downsize"],
                            max_size_mb=config["max_size_mb"]
                        )
                        
                        # Add this job to the processor
                        processor.add_job(download_config)
                    
                    # Track playlist progress
                    total_jobs = len(video_urls)
                    completed_jobs = 0
                    
                    # Create a thread to monitor batch progress
                    def monitor_playlist_progress():
                        nonlocal completed_jobs
                        while processor.processing and self.is_downloading:
                            # Get current job statuses
                            job_statuses = processor.list_jobs()
                            
                            # Count completed jobs
                            new_completed = sum(1 for job in job_statuses if job["status"] in ["completed", "failed"])
                            
                            if new_completed != completed_jobs:
                                completed_jobs = new_completed
                                
                            # Update progress bar - scale from 20% to 100% since we're already at 20%
                            progress = 0.2 + ((completed_jobs / total_jobs) * 0.8) if total_jobs > 0 else 0.2
                            status_message = f"Processing {completed_jobs}/{total_jobs} videos from playlist"
                            
                            # Update UI
                            self._update_progress(progress, prefix="playlist_")
                            self._update_status(status_message, prefix="playlist_")
                            
                            # Log progress
                            if completed_jobs > 0 and completed_jobs % 5 == 0:
                                self._add_log(f"Playlist progress: {completed_jobs}/{total_jobs} videos processed")
                            
                            time.sleep(1)
                    
                    # Start the monitoring thread
                    monitor_thread = threading.Thread(target=monitor_playlist_progress, daemon=True)
                    monitor_thread.start()
                    
                    # Start batch processing
                    self._update_status("Processing playlist videos...", prefix="playlist_")
                    processor.start_processing()
                    
                    # Wait for processing to complete
                    while processor.processing and self.is_downloading:
                        time.sleep(0.5)  # Check more frequently but let the monitor thread handle updates
                    
                    # Display results
                    processor.display_results()
                    
                    self._update_status(f"Playlist '{playlist_title}' processing complete", prefix="playlist_")
                    self._update_progress(1.0, prefix="playlist_")
            finally:
                # Clean up temp directory
                import shutil
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            
        except Exception as e:
            self._update_status(f"Error: {str(e)}", prefix="playlist_")
            logger.error(f"Playlist processing error: {str(e)}")
            self._update_progress(0.0, prefix="playlist_")
        finally:
            self.is_downloading = False
    
    def _cancel_download(self):
        """Cancel the current download process"""
        if self.is_downloading:
            self.is_downloading = False
            # Note: This doesn't actually stop the thread directly 
            # (would need a more complex implementation with proper thread management)
            self._update_status("Download cancelled")
            self._update_progress(0.0)
    
    def run(self):
        """Run the GUI application"""
        self.setup_gui()
        dpg.start_dearpygui()
        dpg.destroy_context()


def main():
    """Main entry point for the GUI application"""
    app = YouTubeDownloaderGUI()
    app.run()


if __name__ == "__main__":
    main()
