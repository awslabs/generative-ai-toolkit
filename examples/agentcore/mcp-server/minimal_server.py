#!/usr/bin/env python3
"""
Minimal MCP Server for AgentCore Runtime Testing

This is a minimal server that just starts up and logs messages to verify
that the AgentCore runtime container can start and send logs to CloudWatch.
"""

import logging
import time
import sys
import signal
import os
from datetime import datetime

# Configure logging to output to stdout (which goes to CloudWatch)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True


def main():
    """Main server loop that just logs messages periodically."""
    global shutdown_requested

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("=== Minimal MCP Server Starting ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Environment variables:")
    for key, value in os.environ.items():
        if key.startswith(("AWS_", "BEDROCK_", "MCP_", "PORT", "HOST")):
            logger.info(f"  {key}={value}")

    logger.info("Server started successfully - entering main loop")

    counter = 0
    while not shutdown_requested:
        counter += 1
        logger.info(
            f"Heartbeat #{counter} - Server is running at {datetime.now().isoformat()}"
        )

        # Log some system info periodically
        if counter % 10 == 0:
            logger.info(f"Memory usage check #{counter//10}")
            logger.info(f"Process ID: {os.getpid()}")

        # Sleep for 30 seconds between heartbeats
        for _ in range(30):
            if shutdown_requested:
                break
            time.sleep(1)

    logger.info("=== Minimal MCP Server Shutting Down ===")
    logger.info("Shutdown completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        sys.exit(1)
