import re
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from pydantic import ValidationError

from models import VideoDownloadConfig
from downloader import YouTubeDownloader
from batch import BatchProcessor


def parse_arguments():
    """Parse command line arguments with input validation"""
    parser = argparse.ArgumentParser(
        description="YouTube Video/Audio Downloader & Optimizer"
    )
    
    # Import sys for checking command line arguments
    import sys
    
    # Detect if the command looks like a direct URL use (backward compatibility)
    is_direct_url = len(sys.argv) > 1 and (
        sys.argv[1].startswith("http") or 
        sys.argv[1].startswith("www.") or
        "youtube.com" in sys.argv[1] or
        "youtu.be" in sys.argv[1]
    )
    
    if is_direct_url:
        # Handle old-style command format (direct URL)
        parser.add_argument(
            "url", help="YouTube URL to download"
        )
        parser.set_defaults(mode="download")
        
        # No need for subparsers in backward compatibility mode
        # Store the main parser for adding common arguments later
        parsers_for_common_args = [parser]
        
    else:
        # Create subparsers for different modes
        subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
        
        # Single download mode
        single_parser = subparsers.add_parser("download", help="Download a single video")
        single_parser.add_argument(
            "url", nargs="?", help="YouTube URL to download (if not provided, will prompt)"
        )
        
        # Batch mode
        batch_parser = subparsers.add_parser("batch", help="Process multiple videos in batch")
        batch_parser.add_argument(
            "-i", "--input-file",
            type=str,
            required=True,
            help="Input file containing URLs (one per line)"
        )
        batch_parser.add_argument(
            "-w", "--workers",
            type=int,
            default=2,
            help="Number of parallel workers (default: 2)"
        )
        
        # Store parsers for adding common arguments
        parsers_for_common_args = [single_parser, batch_parser]
    
    # Common arguments for all parsers
    for p in parsers_for_common_args:
        p.add_argument(
            "-o", "--output-dir",
            type=str,
            help="Output directory for downloaded files"
        )
        p.add_argument(
            "--quiet",
            action="store_true",
            help="Reduce output verbosity"
        )
        p.add_argument(
            "-q", "--quality",
            type=str,
            choices=["high", "medium", "low"],
            default="high",
            help="Quality preset (default: high)"
        )
        p.add_argument(
            "-f", "--format",
            type=str,
            help="Output video format (e.g., mp4, mkv, webm)"
        )
        p.add_argument(
            "--video-format",
            type=str,
            default="mp4",
            help="Video format (default: mp4)"
        )
        p.add_argument(
            "--audio-format",
            type=str,
            default="mp3",
            choices=["mp3", "aac", "ogg", "flac"],
            help="Audio format (default: mp3)"
        )
        p.add_argument(
            "--audio-quality",
            type=str,
            default="320k",
            help="Audio quality bitrate (default: 320k)"
        )
        p.add_argument(
            "--video-only",
            action="store_true",
            help="Only download and optimize video"
        )
        p.add_argument(
            "--audio-only",
            action="store_true",
            help="Only download and extract audio"
        )
        p.add_argument(
            "--downsize",
            action="store_true",
            help="Downsize video to a maximum file size (8MB by default)"
        )
        p.add_argument(
            "--max-size",
            type=int,
            default=8,
            help="Maximum file size in MB when downsizing (default: 8MB)"
        )
    
    # If no mode is specified, default to download mode
    parser.set_defaults(mode="download")

    args = parser.parse_args()

    # Additional validation of output directory
    if args.output_dir:
        output_path = Path(args.output_dir)
        # Resolve to absolute path to prevent path traversal
        output_path = output_path.resolve()
        # Convert back to string for the args
        args.output_dir = str(output_path)

    return args


def run_downloader(args) -> Dict[str, Any]:
    """Run the downloader with the provided arguments"""
    # Set up console
    console = Console()
    
    # Determine log level from arguments
    log_level = logging.WARNING if args.quiet else logging.INFO
    
    # Handle different modes
    if args.mode == "batch":
        return run_batch_mode(args, console, log_level)
    else:  # download mode (single video or playlist)
        # Check if URL is a playlist before proceeding
        if args.url and "playlist" in args.url:
            return run_playlist_mode(args, console, log_level)
        console.print(
            Panel("YouTube Video Downloader", title="🎥 Welcome", border_style="blue")
        )
        
        # Get URL from arguments or prompt with validation
        url = args.url
        if not url:
            url = console.input("Enter YouTube Video URL: ")
            # Basic validation at input time
            youtube_pattern = r'^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[a-zA-Z0-9_-]{11}.*$'
            if not re.match(youtube_pattern, url):
                console.print(Panel("Invalid YouTube URL format", title="Error", border_style="red"))
                return {"success": False, "error": "Invalid YouTube URL format"}

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
        # Check if format is an audio format and set mode accordingly
        audio_only = args.audio_only
        video_only = args.video_only
        
        # Common audio formats to detect from -f parameter
        audio_formats = ["mp3", "flac", "aac", "ogg", "m4a", "wav"]
        
        # If format is specified and it's an audio format, assume audio-only download
        if args.format and args.format.lower() in audio_formats:
            audio_format = args.format
            audio_only = True
            video_format = "mp4"  # Default fallback, won't be used for audio-only
        else:
            # Normal operation
            video_format = args.format or args.video_format
            audio_format = args.audio_format

        # Simple format validation
        format_pattern = r'^[a-zA-Z0-9]+$'
        if video_format and not re.match(format_pattern, video_format):
            console.print(Panel("Invalid video format - use only letters and numbers", title="Error", border_style="red"))
            return {"success": False, "error": "Invalid video format"}

        if audio_format and not re.match(format_pattern, audio_format):
            console.print(Panel("Invalid audio format - use only letters and numbers", title="Error", border_style="red"))
            return {"success": False, "error": "Invalid audio format"}

        config = VideoDownloadConfig(
            url=url,
            video_format=video_format,
            audio_format=audio_format,
            audio_quality=args.audio_quality,
            output_dir=output_dir,
            video_only=video_only,
            audio_only=audio_only,
            quality=args.quality,
            format=args.format,
            downsize=args.downsize,
            max_size_mb=args.max_size
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
        return {"success": False, "error": "Configuration validation failed"}

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

    return result


def run_playlist_mode(args, console, log_level) -> Dict[str, Any]:
    """Extract videos from a playlist and process them in batch"""
    from yt_dlp import YoutubeDL
    import tempfile
    
    console.print(
        Panel("YouTube Playlist Downloader", title="🎬 Playlist Processing", border_style="green")
    )
    
    # Set up output directory with security validation
    if args.output_dir:
        output_dir = Path(args.output_dir)
        # Resolve to absolute path and validate
        output_dir = output_dir.resolve()
    else:
        output_dir = Path.cwd() / "playlist_output"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Output directory: {output_dir}")
    
    # Create a temporary file to store the extracted URLs
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as temp_file:
        temp_file_path = Path(temp_file.name)
        
        # Extract video URLs from the playlist
        console.print("Extracting videos from playlist...")
        
        # Set up yt-dlp options for playlist extraction
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,  # Don't download any videos, just extract info
            'skip_download': True
        }
        
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(args.url, download=False)
                
                if 'entries' in info:
                    # Write each video URL to the temp file
                    count = 0
                    for entry in info['entries']:
                        if entry and 'url' in entry:
                            video_id = entry.get('id')
                            if video_id:
                                video_url = f"https://www.youtube.com/watch?v={video_id}"
                                temp_file.write(f"{video_url}\n")
                                count += 1
                    
                    console.print(f"Found {count} videos in playlist")
                    temp_file.flush()  # Ensure all data is written to the file
                else:
                    console.print(Panel("No videos found in playlist", title="Error", border_style="red"))
                    return {"success": False, "error": "No videos found in playlist"}
        except Exception as e:
            console.print(Panel(f"Error extracting playlist: {str(e)}", title="Error", border_style="red"))
            return {"success": False, "error": f"Error extracting playlist: {str(e)}"}
    
    # Now that we have the URLs in a file, use the batch mode to process them
    # Create a modified args object with the temp file as input
    import argparse
    modified_args = argparse.Namespace(
        mode="batch",
        input_file=str(temp_file_path),
        workers=2,  # Default to 2 workers for playlists
        output_dir=args.output_dir,
        quiet=args.quiet,
        quality=args.quality,
        format=args.format,
        video_format=args.video_format,
        audio_format=args.audio_format,
        audio_quality=args.audio_quality,
        video_only=args.video_only,
        audio_only=args.audio_only,
        downsize=args.downsize,
        max_size=args.max_size
    )
    
    # Run batch processing on the playlist videos
    result = run_batch_mode(modified_args, console, log_level)
    
    # Clean up the temporary file
    try:
        temp_file_path.unlink()
    except Exception:
        pass
    
    return result


def run_batch_mode(args, console, log_level) -> Dict[str, Any]:
    """Run the downloader in batch mode"""
    console.print(
        Panel("YouTube Batch Downloader", title="🎬 Batch Processing", border_style="green")
    )
    
    # Set up output directory with security validation
    if args.output_dir:
        output_dir = Path(args.output_dir)
        # Resolve to absolute path and validate
        output_dir = output_dir.resolve()
    else:
        output_dir = Path.cwd() / "batch_output"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Output directory: {output_dir}")
    
    # Read URLs from input file
    input_file = Path(args.input_file)
    if not input_file.exists():
        console.print(Panel(f"Input file not found: {input_file}", title="Error", border_style="red"))
        return {"success": False, "error": "Input file not found"}
    
    try:
        with open(input_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
    except Exception as e:
        console.print(Panel(f"Error reading input file: {str(e)}", title="Error", border_style="red"))
        return {"success": False, "error": f"Error reading input file: {str(e)}"}
    
    if not urls:
        console.print(Panel("No URLs found in input file", title="Error", border_style="red"))
        return {"success": False, "error": "No URLs found in input file"}
    
    console.print(f"Found {len(urls)} URLs to process")
    
    # Create batch processor
    processor = BatchProcessor(
        base_output_dir=output_dir,
        max_workers=args.workers,
        log_level=log_level
    )
    
    # Add jobs to queue
    job_ids = []
    for url in urls:
        try:
            # Validate URL
            youtube_pattern = r'^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[a-zA-Z0-9_-]{11}.*$'
            if not re.match(youtube_pattern, url):
                console.print(f"Skipping invalid URL: {url}")
                continue
                
            # Create config for this URL
            config = VideoDownloadConfig(
                url=url,
                video_format=args.video_format,
                audio_format=args.audio_format,
                audio_quality=args.audio_quality,
                video_only=args.video_only,
                audio_only=args.audio_only,
                quality=args.quality,
                format=args.format,
                downsize=args.downsize,
                max_size_mb=args.max_size
            )
            
            # Add job to processor
            job_id = processor.add_job(config)
            job_ids.append(job_id)
            
        except Exception as e:
            console.print(f"Error adding job for {url}: {str(e)}")
    
    # Start processing
    processor.start_processing()
    
    # Wait for processing to complete with progress updates
    console.print("Processing videos in parallel... Press Ctrl+C to stop (jobs will continue in background)")
    try:
        while processor.processing:
            time.sleep(1)
            
            # Get job statuses
            statuses = [processor.get_job_status(job_id) for job_id in job_ids]
            completed = sum(1 for s in statuses if s["status"] in ["completed", "failed"])
            
            console.print(f"Progress: {completed}/{len(job_ids)} jobs completed", end="\r")
    
    except KeyboardInterrupt:
        console.print("\nInterrupted, but jobs will continue in the background.")
    
    # Display results
    processor.display_results()
    
    return {"success": True, "job_count": len(job_ids)}
