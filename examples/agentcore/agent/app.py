"""
Copyright 2024 Amazon.com, Inc. and its affiliates. All Rights Reserved.

Licensed under the Amazon Software License (the "License").
You may not use this file except in compliance with the License.
A copy of the License is located at

  http://aws.amazon.com/asl/

or in the "license" file accompanying this file. This file is distributed
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
express or implied. See the License for the specific language governing
permissions and limitations under the License.
"""

"""
AgentCore HTTP Handler

This module implements the HTTP endpoints required for AgentCore integration:
- POST /invocations: Main agent invocation endpoint
- GET /ping: Health check endpoint

The handler integrates with the WeatherAgent and provides proper request/response
handling for the AgentCore Runtime environment.
"""

import os
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any

from flask import Flask, request, jsonify
from pydantic import ValidationError

from weather_agent import WeatherAgent
from models import AgentCoreRequest, AgentCoreResponse, HealthCheckResponse


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create Flask application
app = Flask(__name__)

# Global agent instance
weather_agent: WeatherAgent = None


def initialize_agent():
    """Initialize the weather agent with configuration from environment variables."""
    global weather_agent

    try:
        # Get configuration from environment variables
        model_id = os.getenv(
            "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
        mcp_server_endpoint = os.getenv("MCP_SERVER_ENDPOINT")
        region = os.getenv("AWS_REGION", "us-east-1")
        max_tokens = int(os.getenv("MAX_TOKENS", "4096"))
        temperature = float(os.getenv("TEMPERATURE", "0.1"))

        logger.info(f"Initializing WeatherAgent with model: {model_id}")
        logger.info(f"MCP Server Endpoint: {mcp_server_endpoint}")
        logger.info(f"AWS Region: {region}")

        # Create the weather agent
        weather_agent = WeatherAgent(
            model_id=model_id,
            mcp_server_endpoint=mcp_server_endpoint,
            region=region,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        logger.info("WeatherAgent initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize WeatherAgent: {str(e)}")
        raise


@app.route("/invocations", methods=["POST"])
def handle_invocations():
    """
    Handle AgentCore invocation requests.

    This endpoint receives JSON requests from AgentCore Runtime and processes
    them using the WeatherAgent, returning appropriate responses.
    """
    try:
        # Log the incoming request
        logger.info("Received invocation request")

        # Ensure agent is initialized
        if weather_agent is None:
            logger.error("WeatherAgent not initialized")
            return (
                jsonify(
                    {
                        "error": "Agent not initialized",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                ),
                500,
            )

        # Parse request JSON
        request_data = request.get_json()
        if not request_data:
            logger.error("No JSON data in request")
            return (
                jsonify(
                    {
                        "error": "No JSON data provided",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                ),
                400,
            )

        logger.debug(f"Request data: {request_data}")

        # Validate request using Pydantic model
        try:
            agent_request = AgentCoreRequest(**request_data)
        except ValidationError as e:
            logger.error(f"Request validation failed: {str(e)}")
            return (
                jsonify(
                    {
                        "error": f"Invalid request format: {str(e)}",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                ),
                400,
            )

        # Process the request with the weather agent
        try:
            # Run the async process_request method
            response = asyncio.run(weather_agent.process_request(agent_request))

            # Convert response to dictionary
            response_dict = response.model_dump()

            logger.info(
                f"Successfully processed request for session {agent_request.session_id}"
            )
            logger.debug(f"Response: {response_dict}")

            return jsonify(response_dict), 200

        except Exception as e:
            logger.error(f"Error processing agent request: {str(e)}")

            # Return error response in AgentCore format
            error_response = AgentCoreResponse(
                response=f"I apologize, but I encountered an error: {str(e)}",
                session_id=agent_request.session_id,
                metadata={"error": str(e), "timestamp": datetime.utcnow().isoformat()},
            )

            return jsonify(error_response.model_dump()), 500

    except Exception as e:
        logger.error(f"Unexpected error in invocations handler: {str(e)}")
        return (
            jsonify(
                {
                    "error": f"Internal server error: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            500,
        )


@app.route("/ping", methods=["GET"])
def handle_ping():
    """
    Handle health check requests from AgentCore.

    This endpoint is used by AgentCore to verify that the agent container
    is healthy and ready to receive requests.
    """
    try:
        logger.debug("Received ping request")

        # Basic health check
        health_status = "healthy"

        # Check if agent is initialized
        if weather_agent is None:
            health_status = "unhealthy"
            logger.warning("Ping check failed: WeatherAgent not initialized")
        else:
            # Get detailed health information from the agent
            try:
                agent_health = weather_agent.health_check()
                if agent_health.get("status") != "healthy":
                    health_status = "unhealthy"
            except Exception as e:
                logger.error(f"Agent health check failed: {str(e)}")
                health_status = "unhealthy"

        # Create health check response
        health_response = HealthCheckResponse(
            status=health_status,
            version="1.0.0",
            timestamp=datetime.utcnow().isoformat(),
        )

        status_code = 200 if health_status == "healthy" else 503

        logger.debug(f"Ping response: {health_response.model_dump()}")

        return jsonify(health_response.model_dump()), status_code

    except Exception as e:
        logger.error(f"Error in ping handler: {str(e)}")

        # Return unhealthy status
        error_response = HealthCheckResponse(
            status="unhealthy", version="1.0.0", timestamp=datetime.utcnow().isoformat()
        )

        return jsonify(error_response.model_dump()), 503


@app.route("/health", methods=["GET"])
def handle_health():
    """
    Additional health check endpoint for debugging and monitoring.

    This provides more detailed health information than the /ping endpoint.
    """
    try:
        logger.debug("Received detailed health check request")

        # Get basic Flask app info
        health_info = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "flask_debug": app.debug,
            "environment": {
                "BEDROCK_MODEL_ID": os.getenv("BEDROCK_MODEL_ID", "not_set"),
                "MCP_SERVER_ENDPOINT": os.getenv("MCP_SERVER_ENDPOINT", "not_set"),
                "AWS_REGION": os.getenv("AWS_REGION", "not_set"),
            },
        }

        # Add agent health information if available
        if weather_agent is not None:
            try:
                agent_health = weather_agent.health_check()
                health_info["agent"] = agent_health
            except Exception as e:
                health_info["agent"] = {"error": str(e)}
                health_info["status"] = "degraded"
        else:
            health_info["agent"] = {"status": "not_initialized"}
            health_info["status"] = "unhealthy"

        status_code = 200 if health_info["status"] in ["healthy", "degraded"] else 503

        return jsonify(health_info), status_code

    except Exception as e:
        logger.error(f"Error in health handler: {str(e)}")
        return (
            jsonify(
                {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            503,
        )


@app.route("/", methods=["GET"])
def handle_root():
    """Root endpoint that provides basic service information."""
    return jsonify(
        {
            "service": "weather-agent",
            "version": "1.0.0",
            "status": "running",
            "timestamp": datetime.utcnow().isoformat(),
            "endpoints": {
                "invocations": "POST /invocations",
                "ping": "GET /ping",
                "health": "GET /health",
            },
        }
    )


@app.errorhandler(404)
def handle_not_found(error):
    """Handle 404 errors."""
    return (
        jsonify(
            {
                "error": "Endpoint not found",
                "message": "The requested endpoint does not exist",
                "timestamp": datetime.utcnow().isoformat(),
            }
        ),
        404,
    )


@app.errorhandler(405)
def handle_method_not_allowed(error):
    """Handle 405 errors."""
    return (
        jsonify(
            {
                "error": "Method not allowed",
                "message": "The HTTP method is not allowed for this endpoint",
                "timestamp": datetime.utcnow().isoformat(),
            }
        ),
        405,
    )


@app.errorhandler(500)
def handle_internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return (
        jsonify(
            {
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "timestamp": datetime.utcnow().isoformat(),
            }
        ),
        500,
    )


def create_app():
    """
    Application factory function.

    Returns:
        Configured Flask application
    """
    # Initialize the agent
    initialize_agent()

    return app


if __name__ == "__main__":
    try:
        # Initialize the agent
        initialize_agent()

        # Get server configuration
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8080"))
        debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"

        logger.info(f"Starting Weather Agent HTTP server on {host}:{port}")
        logger.info(f"Debug mode: {debug}")

        # Start the Flask application
        app.run(host=host, port=port, debug=debug, threaded=True)

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise
    finally:
        # Cleanup
        if weather_agent:
            weather_agent.cleanup()
