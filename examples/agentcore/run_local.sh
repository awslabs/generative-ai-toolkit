#!/bin/bash

# Copyright 2024 Amazon.com, Inc. and its affiliates. All Rights Reserved.
#
# Licensed under the Amazon Software License (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#   http://aws.amazon.com/asl/
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Local Development Helper Script for AgentCore Integration
#
# This script provides convenient commands for local development and testing
# of the weather agent with docker-compose.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
AGENT_ENDPOINT="http://localhost:8080"
MCP_ENDPOINT="http://localhost:8000"

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if docker-compose is available
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null && ! command -v docker &> /dev/null; then
        log_error "Docker and docker-compose are required but not installed"
        exit 1
    fi
    
    # Use docker compose (newer) or docker-compose (legacy)
    if docker compose version &> /dev/null; then
        DOCKER_COMPOSE="docker compose"
    elif docker-compose version &> /dev/null; then
        DOCKER_COMPOSE="docker-compose"
    else
        log_error "Neither 'docker compose' nor 'docker-compose' is available"
        exit 1
    fi
    
    log_info "Using: $DOCKER_COMPOSE"
}

# Start the services
start_services() {
    log_info "Starting AgentCore local development environment..."
    
    # Build and start services
    $DOCKER_COMPOSE -f $COMPOSE_FILE up --build -d
    
    log_success "Services started successfully!"
    log_info "Agent endpoint: $AGENT_ENDPOINT"
    log_info "MCP server endpoint: $MCP_ENDPOINT"
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 10
    
    # Check health
    check_health
}

# Stop the services
stop_services() {
    log_info "Stopping AgentCore local development environment..."
    $DOCKER_COMPOSE -f $COMPOSE_FILE down
    log_success "Services stopped successfully!"
}

# Restart the services
restart_services() {
    log_info "Restarting AgentCore local development environment..."
    stop_services
    start_services
}

# Check service health
check_health() {
    log_info "Checking service health..."
    
    # Check MCP server
    if curl -s -f "$MCP_ENDPOINT/health" > /dev/null; then
        log_success "MCP server is healthy"
    else
        log_warning "MCP server health check failed"
    fi
    
    # Check agent
    if curl -s -f "$AGENT_ENDPOINT/ping" > /dev/null; then
        log_success "Weather agent is healthy"
    else
        log_warning "Weather agent health check failed"
    fi
}

# Show service logs
show_logs() {
    local service=$1
    if [ -z "$service" ]; then
        log_info "Showing logs for all services..."
        $DOCKER_COMPOSE -f $COMPOSE_FILE logs -f
    else
        log_info "Showing logs for service: $service"
        $DOCKER_COMPOSE -f $COMPOSE_FILE logs -f "$service"
    fi
}

# Run a quick test
run_test() {
    log_info "Running quick test against local agent..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required to run tests"
        exit 1
    fi
    
    # Install requests if not available
    python3 -c "import requests" 2>/dev/null || {
        log_info "Installing requests library..."
        pip3 install requests
    }
    
    # Run a quick test using the test_local.py script
    python3 tests/test_local.py --endpoint "$AGENT_ENDPOINT" --test-type basic
}

# Run comprehensive tests
run_full_tests() {
    log_info "Running comprehensive tests against local agent..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required to run tests"
        exit 1
    fi
    
    # Install requests if not available
    python3 -c "import requests" 2>/dev/null || {
        log_info "Installing requests library..."
        pip3 install requests
    }
    
    # Run all tests
    python3 tests/test_local.py --endpoint "$AGENT_ENDPOINT" --test-type all
}

# Show service status
show_status() {
    log_info "Service status:"
    $DOCKER_COMPOSE -f $COMPOSE_FILE ps
}

# Clean up everything
cleanup() {
    log_info "Cleaning up AgentCore local environment..."
    $DOCKER_COMPOSE -f $COMPOSE_FILE down -v --remove-orphans
    docker system prune -f
    log_success "Cleanup completed!"
}

# Show help
show_help() {
    echo "AgentCore Local Development Helper"
    echo "=================================="
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start       Start the local development environment"
    echo "  stop        Stop the local development environment"
    echo "  restart     Restart the local development environment"
    echo "  status      Show service status"
    echo "  health      Check service health"
    echo "  logs [svc]  Show logs (optionally for specific service)"
    echo "  test        Run a quick test"
    echo "  test-all    Run comprehensive tests"
    echo "  cleanup     Clean up all containers and volumes"
    echo "  help        Show this help message"
    echo ""
    echo "Services:"
    echo "  weather-agent  - Weather agent container (port 8080)"
    echo "  mcp-server     - MCP tools server container (port 8000)"
    echo ""
    echo "Endpoints:"
    echo "  Agent:     $AGENT_ENDPOINT"
    echo "  MCP:       $MCP_ENDPOINT"
}

# Main script logic
main() {
    check_docker_compose
    
    case "${1:-help}" in
        start)
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        status)
            show_status
            ;;
        health)
            check_health
            ;;
        logs)
            show_logs "$2"
            ;;
        test)
            run_test
            ;;
        test-all)
            run_full_tests
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"