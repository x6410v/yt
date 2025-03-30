# YouTube Video Downloader

## Overview

A robust Python command-line utility for downloading, optimizing, and extracting audio from YouTube videos using modern Python libraries and tools.

## Features

- YouTube URL validation
- High-quality video download with yt-dlp
- Video optimization with FFmpeg
- Quality presets (high, medium, low)
- Flexible output format options
- High-quality audio extraction (up to 320kbps)
- Multiple audio format support (MP3, AAC, OGG, FLAC)
- Secure file hashing for verification
- Comprehensive logging
- Rich console output
- Command-line interface with argparse

## Prerequisites

- Python 3.9+
- FFmpeg installed system-wide
- yt-dlp library
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository
2. Create a virtual environment
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python youtube_downloader.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

### Command-line Arguments

```
usage: youtube_downloader.py [-h] [-o OUTPUT_DIR] [--quiet] [-q {high,medium,low}]
                             [-f FORMAT] [--video-format VIDEO_FORMAT]
                             [--audio-format {mp3,aac,ogg,flac}]
                             [--audio-quality AUDIO_QUALITY] [--video-only]
                             [--audio-only]
                             [url]

YouTube Video/Audio Downloader & Optimizer

positional arguments:
  url                   YouTube URL to download (if not provided, will prompt)

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output directory for downloaded files
  --quiet               Reduce output verbosity
  -q {high,medium,low}, --quality {high,medium,low}
                        Quality preset (default: high)
  -f FORMAT, --format FORMAT
                        Output video format (e.g., mp4, mkv, webm)
  --video-format VIDEO_FORMAT
                        Video format (default: mp4)
  --audio-format {mp3,aac,ogg,flac}
                        Audio format (default: mp3)
  --audio-quality AUDIO_QUALITY
                        Audio quality bitrate (default: 320k)
  --video-only          Only download and optimize video
  --audio-only          Only download and extract audio
```

### Quality Presets

The tool offers three quality presets that affect both video and audio quality:

- **high**: Best video resolution, 320kbps audio, slow encoding for best compression
- **medium**: 720p video, 192kbps audio, medium encoding speed
- **low**: 480p video, 128kbps audio, fast encoding

### Examples

Download and process a video at high quality:

```bash
python youtube_downloader.py https://www.youtube.com/watch?v=dQw4w9WgXcQ -q high
```

Download in medium quality and save as MKV:

```bash
python youtube_downloader.py https://www.youtube.com/watch?v=dQw4w9WgXcQ -q medium -f mkv
```

Download audio only in FLAC format:

```bash
python youtube_downloader.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --audio-only --audio-format flac
```

Download to a specific directory:

```bash
python youtube_downloader.py https://www.youtube.com/watch?v=dQw4w9WgXcQ -o ~/Downloads/yt-videos
```

## Output

The script will:

1. Create an output directory (either specified or a temporary one)
2. Download the video in the requested quality
3. Create an optimized version of the video in the specified format
4. Extract audio in the requested format and quality
5. Generate a JSON log file with details and file hashes

## Safety & Privacy

- Videos are stored in the specified directory
- Temporary files are automatically cleaned up
- Detailed logs are generated for each download
- SHA-256 hashing ensures file integrity

## Dependencies

- yt-dlp
- ffmpeg-python
- rich
- pydantic
- validators

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

## License
