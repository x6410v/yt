#!/usr/bin/env python3
"""
YouTube Video Downloader & Optimizer
Main entry point for the application

Supports both CLI and GUI modes:
- CLI: Default mode, run with command line arguments (see README.md)
- GUI: Run with --gui flag to launch the graphical interface
"""

import sys
from cli import parse_arguments, run_downloader


def main():
    """Main entry point for the application"""
    # Check for GUI mode
    if len(sys.argv) > 1 and sys.argv[1] == "--gui":
        # Remove the --gui flag from arguments
        sys.argv.pop(1)
        # Import GUI module (only when needed to avoid unnecessary dependencies)
        from gui import YouTubeDownloaderGUI
        # Launch GUI
        app = YouTubeDownloaderGUI()
        app.run()
    else:
        # CLI mode
        args = parse_arguments()
        run_downloader(args)


if __name__ == "__main__":
    main()
