#!/usr/bin/env python3
"""
Batch processing module for YouTube Downloader
Allows processing multiple videos in parallel with enhanced security and memory management
"""

import os
import time
import logging
import json
import queue
import threading
import concurrent.futures
import gc
import tracemalloc
import psutil
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
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
        self.logger.info(f"Added job {job_id} to queue: {str(config.url)}")
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
            "url": str(job_info["config"].url),  # Convert HttpUrl to string
            "status": job_info["status"],
            "added": job_info["added"],
            "started": job_info["started"],
            "completed": job_info["completed"]
        } for job_id, job_info in self.jobs.items()]
    
    def process_job(self, job_id: int, config: VideoDownloadConfig) -> Dict[str, Any]:
        """Process a single job with enhanced security and error handling"""
        self.jobs[job_id]["status"] = "processing"
        self.jobs[job_id]["started"] = time.time()
        
        try:
            # Security: Validate URL before processing
            # Check for empty URL
            if not config.url:
                raise ValueError("Empty URL provided")
                
            # Convert URL to string for validation
            url_str = str(config.url)
            if not url_str:
                raise ValueError("Invalid or missing URL")
                
            # YouTube URL validation 
            if not ("youtube.com" in url_str or "youtu.be" in url_str):
                self.logger.warning(f"URL may not be a valid YouTube URL: {url_str}")
                
            # Convert HttpUrl to string and validate
            url_str = str(config.url)
            
            # Security: Sanitize URL for potentially dangerous characters
            if any(char in url_str for char in [';', '|', '`', '$', '\\']):
                raise ValueError(f"Security: URL contains potentially unsafe characters: {url_str}")
                
            # Note: We removed '&' from the unsafe characters list since it's common in YouTube URLs
            
            # Create a dedicated output directory for this job
            job_dir_name = f"job_{job_id}_{int(time.time())}"
            job_output_dir = self.base_output_dir / job_dir_name
            
            # Security: Validate output directory path
            try:
                job_output_dir.mkdir(parents=True, exist_ok=True)
                if not job_output_dir.exists() or not os.access(job_output_dir, os.W_OK):
                    raise PermissionError(f"Cannot write to output directory: {job_output_dir}")
            except (PermissionError, OSError) as e:
                self.logger.error(f"Directory creation error for job {job_id}: {str(e)}")
                raise
                
            # Update config with the job-specific output directory
            config.output_dir = job_output_dir
            
            # Initialize the downloader with a dedicated logger
            job_log_path = job_output_dir / f"job_{job_id}.log"
            job_file_handler = logging.FileHandler(job_log_path)
            job_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            
            # Initialize the downloader with memory and resource limits
            downloader = YouTubeDownloader(
                config=config,
                log_level=self.log_level,
                extra_handlers=[job_file_handler]
            )
            
            # Set a timeout for the job to prevent hanging
            start_time = time.time()
            max_job_time = 600  # 10 minutes max per job (reduced from 30 minutes)
            
            # Create a timeout mechanism for the job processing
            job_thread = None
            job_result = {"success": False, "video_path": None, "error": None}
            
            def process_with_timeout():
                try:
                    # Process the video
                    result = downloader.process_video()
                    job_result.update(result)
                except Exception as e:
                    job_result["error"] = str(e)
            
            # Start processing in a separate thread
            job_thread = threading.Thread(target=process_with_timeout)
            job_thread.daemon = True
            job_thread.start()
            
            # Wait for processing with timeout
            job_thread.join(timeout=max_job_time)
            
            # Check if processing completed or timed out
            if job_thread.is_alive():
                self.logger.error(f"Job {job_id} timed out after {max_job_time} seconds")
                result = {
                    "success": False,
                    "error": f"Job timed out after {max_job_time} seconds",
                    "error_type": "TimeoutError"
                }
            else:
                # Use the result from the thread
                result = job_result
            
            # Memory management: Clean up large objects
            if hasattr(downloader, 'cleanup'):
                downloader.cleanup()
                
            # Job duration validation
            job_duration = time.time() - start_time
            if job_duration > max_job_time:
                self.logger.warning(f"Job {job_id} took longer than expected: {job_duration:.2f}s")
            
            # Update job status with enhanced metadata
            self.jobs[job_id]["status"] = "completed" if result["success"] else "failed"
            self.jobs[job_id]["completed"] = time.time()
            self.jobs[job_id]["duration"] = job_duration
            self.jobs[job_id]["log_file"] = str(job_log_path)
            
            # Store only essential result data to conserve memory
            if "video_path" in result:
                # Convert Path objects to strings to avoid serialization issues
                result["video_path"] = str(result["video_path"])
                
            # Store compressed result data
            self.jobs[job_id]["result"] = result
            
            return result
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error processing job {job_id}: {error_msg}")
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["completed"] = time.time()
            self.jobs[job_id]["result"] = {
                "success": False,
                "error": error_msg,
                "error_type": type(e).__name__
            }
            
            # Log detailed stack trace
            import traceback
            self.logger.error(f"Detailed error for job {job_id}: {traceback.format_exc()}")
            
            return {"success": False, "error": error_msg}
    
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
        """Process the queue using ThreadPoolExecutor with enhanced resource management"""
        try:
            # Set up memory monitoring
            memory_usage_start = self._get_memory_usage()
            self.logger.info(f"Starting batch processing with memory usage: {memory_usage_start:.2f} MB")
            
            with Progress() as progress:
                # Count total jobs in the queue
                total_jobs = self.queue.qsize()
                if total_jobs == 0:
                    self.logger.warning("No jobs in queue to process")
                    return
                    
                self.logger.info(f"Processing {total_jobs} jobs with {self.max_workers} workers")
                
                # Create a master task to track overall progress
                master_task = progress.add_task("[bold blue]Processing jobs...", total=total_jobs)
                
                # Create a dict to track individual job tasks
                job_tasks = {}
                
                # Create a list to hold active futures
                active_futures = []
                completed_jobs = 0
                failed_jobs = 0
                
                # Limit batch size to prevent excessive memory usage
                batch_size = min(20, total_jobs)  # Process in batches of 20 jobs max
                self.logger.info(f"Using batch size of {batch_size} for job processing")
                
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Process jobs in batches to limit memory usage
                    while not self.queue.empty() or active_futures:
                        # Fill the active futures list up to batch_size
                        while len(active_futures) < batch_size and not self.queue.empty():
                            try:
                                job_id, config = self.queue.get(block=False)
                                
                                # Add a task for this job
                                # Convert HttpUrl to string before slicing
                                url_str = str(config.url)
                                job_tasks[job_id] = progress.add_task(
                                    f"[cyan]Job {job_id}: {url_str[:50]}...",  # Truncate URL for display
                                    total=100
                                )
                                
                                # Submit the job
                                future = executor.submit(self.process_job, job_id, config)
                                active_futures.append((future, job_id, job_tasks[job_id]))
                                
                                # Periodically check memory usage
                                if len(active_futures) % 5 == 0:
                                    current_memory = self._get_memory_usage()
                                    if current_memory > memory_usage_start * 1.5:  # If memory increased by 50%
                                        self.logger.warning(f"Memory usage increased to {current_memory:.2f} MB")
                            except queue.Empty:
                                break
                        
                        # Wait for at least one future to complete
                        if active_futures:
                            done_futures, _ = concurrent.futures.wait(
                                [f[0] for f in active_futures],
                                timeout=1.0,
                                return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            
                            # Process completed futures
                            still_active = []
                            for future, job_id, task_id in active_futures:
                                if future in done_futures:
                                    try:
                                        # Get result
                                        result = future.result()
                                        
                                        # Mark job task as completed
                                        progress.update(task_id, completed=100)
                                        progress.update(
                                            task_id, 
                                            description=f"[green]Job {job_id}: COMPLETED"
                                        )
                                        
                                        # Update master progress
                                        progress.update(master_task, advance=1)
                                        completed_jobs += 1
                                        
                                        # Collect garbage to free memory
                                        if completed_jobs % 5 == 0:
                                            self._collect_garbage()
                                        
                                    except Exception as e:
                                        self.logger.error(f"Error in job {job_id}: {str(e)}")
                                        
                                        # Mark job task as failed
                                        progress.update(task_id, description=f"[red]Job {job_id}: FAILED")
                                        progress.update(task_id, completed=100)
                                        
                                        # Update master progress
                                        progress.update(master_task, advance=1)
                                        failed_jobs += 1
                                else:
                                    still_active.append((future, job_id, task_id))
                                    
                            # Update active futures list
                            active_futures = still_active
                
                # Final memory usage check
                memory_usage_end = self._get_memory_usage()
                memory_diff = memory_usage_end - memory_usage_start
                self.logger.info(
                    f"Batch processing complete. Memory usage: {memory_usage_end:.2f} MB "
                    f"(change: {memory_diff:+.2f} MB)"
                )
                
                # Final stats
                self.logger.info(f"Completed: {completed_jobs}, Failed: {failed_jobs}")
        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}")
        finally:
            self.processing = False
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            # Convert to MB for readability
            return memory_info.rss / (1024 * 1024)
        except Exception as e:
            self.logger.error(f"Error getting memory usage: {str(e)}")
            return 0.0
    
    def _collect_garbage(self) -> None:
        """Force garbage collection to free memory"""
        try:
            # Get memory before collection
            mem_before = self._get_memory_usage()
            
            # Run garbage collection
            collected = gc.collect()
            
            # Get memory after collection
            mem_after = self._get_memory_usage()
            mem_diff = mem_before - mem_after
            
            if collected > 0 or mem_diff > 10:  # If significant change (>10MB)
                self.logger.debug(f"GC: Collected {collected} objects, freed {mem_diff:.2f} MB")
        except Exception as e:
            self.logger.error(f"Error during garbage collection: {str(e)}")
    
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
