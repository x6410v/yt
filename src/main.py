#!/usr/bin/env python3
"""
YouTube Video Downloader & Optimizer
Main entry point for the application
"""

from cli import parse_arguments, run_downloader


def main():
    """Main entry point for the application"""
    args = parse_arguments()
    run_downloader(args)


if __name__ == "__main__":
    main()
