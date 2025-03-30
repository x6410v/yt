#!/usr/bin/env python3
"""
Batch processing module for YouTube Downloader
Allows processing multiple videos in parallel
"""

import os
import time
import logging
import json
import queue
import threading
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, TaskID

from models import VideoDownloadConfig
from downloader import YouTubeDownloader


class BatchProcessor:
    """Process multiple YouTube videos in parallel"""
    
    def __init__(self, base_output_dir: Path, max_workers: int = 2, log_level: int = logging.INFO):
        self.base_output_dir = base_output_dir
        self.max_workers = max_workers
        self.log_level = log_level
        self.console = Console()
        self.queue = queue.Queue()
        self.results = {}
        self.processing = False
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
        self.logger = logging.getLogger("batch_processor")
        
        # Ensure the base output directory exists
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Keep track of jobs
        self.job_index = 1
        self.jobs = {}
    
    def add_job(self, config: VideoDownloadConfig) -> int:
        """Add a job to the processing queue and return job ID"""
        job_id = self.job_index
        self.job_index += 1
        
        self.jobs[job_id] = {
            "config": config,
            "status": "queued",
            "added": time.time(),
            "started": None,
            "completed": None,
            "result": None
        }
        
        self.queue.put((job_id, config))
        self.logger.info(f"Added job {job_id} to queue: {config.url}")
        return job_id
    
    def get_job_status(self, job_id: int) -> Dict[str, Any]:
        """Get the status of a specific job"""
        if job_id not in self.jobs:
            return {"error": "Job not found"}
        
        return self.jobs[job_id]
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs with their status"""
        return [{
            "job_id": job_id,
            "url": job_info["config"].url,
            "status": job_info["status"],
            "added": job_info["added"],
            "started": job_info["started"],
            "completed": job_info["completed"]
        } for job_id, job_info in self.jobs.items()]
    
    def process_job(self, job_id: int, config: VideoDownloadConfig) -> Dict[str, Any]:
        """Process a single job"""
        self.jobs[job_id]["status"] = "processing"
        self.jobs[job_id]["started"] = time.time()
        
        try:
            # Create a dedicated output directory for this job
            job_output_dir = self.base_output_dir / f"job_{job_id}"
            job_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Update config with the job-specific output directory
            config.output_dir = job_output_dir
            
            # Initialize the downloader
            downloader = YouTubeDownloader(config=config, log_level=self.log_level)
            
            # Process the video
            result = downloader.process_video()
            
            # Update job status
            self.jobs[job_id]["status"] = "completed" if result["success"] else "failed"
            self.jobs[job_id]["completed"] = time.time()
            self.jobs[job_id]["result"] = result
            
            return result
        except Exception as e:
            self.logger.error(f"Error processing job {job_id}: {str(e)}")
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["completed"] = time.time()
            self.jobs[job_id]["result"] = {"success": False, "error": str(e)}
            return {"success": False, "error": str(e)}
    
    def start_processing(self) -> None:
        """Start processing the queue in a separate thread"""
        if self.processing:
            self.logger.warning("Batch processing is already running")
            return
        
        self.processing = True
        processing_thread = threading.Thread(target=self._process_queue)
        processing_thread.daemon = True
        processing_thread.start()
        
        self.console.print(
            Panel(
                f"Started batch processing with {self.max_workers} workers",
                title="Batch Processing",
                border_style="green"
            )
        )
    
    def _process_queue(self) -> None:
        """Process the queue using ThreadPoolExecutor"""
        try:
            with Progress() as progress:
                # Count total jobs in the queue
                total_jobs = self.queue.qsize()
                
                # Create a master task to track overall progress
                master_task = progress.add_task("[bold blue]Processing jobs...", total=total_jobs)
                
                # Create a dict to track individual job tasks
                job_tasks = {}
                
                # Create a list to hold all the futures we submit
                all_futures = []
                
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit all jobs to the executor
                    while not self.queue.empty():
                        job_id, config = self.queue.get()
                        
                        # Add a task for this job
                        job_tasks[job_id] = progress.add_task(
                            f"[cyan]Job {job_id}: {config.url}",
                            total=100
                        )
                        
                        # Submit the job
                        future = executor.submit(self.process_job, job_id, config)
                        all_futures.append((future, job_id, job_tasks[job_id]))
                    
                    # Process completed futures
                    for future, job_id, task_id in all_futures:
                        try:
                            # Wait for the future to complete
                            result = future.result()
                            
                            # Mark job task as completed
                            progress.update(task_id, completed=100)
                            
                            # Update master progress
                            progress.update(master_task, advance=1)
                            
                        except Exception as e:
                            self.logger.error(f"Error in job {job_id}: {str(e)}")
                            
                            # Mark job task as failed
                            progress.update(task_id, description=f"[red]Job {job_id}: FAILED")
                            progress.update(task_id, completed=100)
                            
                            # Update master progress
                            progress.update(master_task, advance=1)
        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}")
        finally:
            self.processing = False
    
    def display_results(self) -> None:
        """Display the results of all jobs in a table"""
        table = Table(title="Batch Processing Results")
        
        table.add_column("Job ID", style="cyan")
        table.add_column("URL", style="blue")
        table.add_column("Status", style="magenta")
        table.add_column("Duration", style="green")
        table.add_column("Result", style="yellow")
        
        for job_id, job_info in self.jobs.items():
            # Calculate duration if job was started
            duration = "N/A"
            if job_info["started"]:
                end_time = job_info["completed"] or time.time()
                duration = f"{end_time - job_info['started']:.2f}s"
            
            # Generate result text
            result_text = "N/A"
            if job_info["result"]:
                if job_info["result"]["success"]:
                    video_path = job_info["result"].get("video_path")
                    if video_path:
                        result_text = f"Video: {Path(video_path).name}"
                else:
                    result_text = f"Error: {job_info['result'].get('error', 'Unknown')}"
            
            table.add_row(
                str(job_id),
                str(job_info["config"].url),
                job_info["status"],
                duration,
                result_text
            )
        
        self.console.print(table)


def create_batch_processor(output_dir: str, max_workers: int = 2) -> BatchProcessor:
    """Create and return a BatchProcessor instance"""
    return BatchProcessor(
        base_output_dir=Path(output_dir).resolve(),
        max_workers=max_workers
    )
