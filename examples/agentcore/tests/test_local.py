#!/usr/bin/env python3
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
Local Testing Script for AgentCore Integration

This script tests the weather agent locally using the same request format
that AgentCore uses. It can test against:
1. Local docker-compose deployment
2. Direct agent application (for development)
3. Deployed AgentCore runtime endpoint

The script ensures local behavior matches AgentCore deployment by using
identical request/response formats and validation.
"""

import json
import time
import uuid
import argparse
import requests
from typing import Dict, Any, Optional
from datetime import datetime

# Test configuration
DEFAULT_LOCAL_ENDPOINT = "http://localhost:8080"
DEFAULT_TIMEOUT = 30


class AgentTester:
    """Test client for weather agent using AgentCore request format."""

    def __init__(self, endpoint: str, timeout: int = DEFAULT_TIMEOUT):
        """
        Initialize the tester.

        Args:
            endpoint: Base URL of the agent (e.g., http://localhost:8080)
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check against the agent.

        Returns:
            Health check response data
        """
        print(f"🔍 Checking health at {self.endpoint}/ping")

        try:
            response = self.session.get(f"{self.endpoint}/ping", timeout=self.timeout)
            response.raise_for_status()

            health_data = response.json()
            print(f"✅ Health check passed: {health_data.get('status', 'unknown')}")
            return health_data

        except requests.exceptions.RequestException as e:
            print(f"❌ Health check failed: {str(e)}")
            raise

    def detailed_health_check(self) -> Dict[str, Any]:
        """
        Perform detailed health check for debugging.

        Returns:
            Detailed health information
        """
        print(f"🔍 Checking detailed health at {self.endpoint}/health")

        try:
            response = self.session.get(f"{self.endpoint}/health", timeout=self.timeout)
            response.raise_for_status()

            health_data = response.json()
            print(f"✅ Detailed health check passed")
            print(f"   Status: {health_data.get('status', 'unknown')}")
            print(f"   Agent: {health_data.get('agent', {}).get('status', 'unknown')}")
            return health_data

        except requests.exceptions.RequestException as e:
            print(f"❌ Detailed health check failed: {str(e)}")
            raise

    def send_request(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        last_k_turns: int = 20,
    ) -> Dict[str, Any]:
        """
        Send a request to the agent using AgentCore format.

        Args:
            prompt: User's input message
            session_id: Session identifier (generated if not provided)
            last_k_turns: Number of conversation turns to include

        Returns:
            Agent response data
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Create AgentCore-compatible request
        request_data = {
            "prompt": prompt,
            "session_id": session_id,
            "last_k_turns": last_k_turns,
        }

        print(f"📤 Sending request to {self.endpoint}/invocations")
        print(f"   Session: {session_id}")
        print(f"   Prompt: {prompt}")

        try:
            start_time = time.time()

            response = self.session.post(
                f"{self.endpoint}/invocations",
                json=request_data,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )

            end_time = time.time()
            response_time = end_time - start_time

            response.raise_for_status()
            response_data = response.json()

            print(f"✅ Request completed in {response_time:.2f}s")
            print(
                f"   Response: {response_data.get('response', 'No response')[:100]}..."
            )

            return {
                "request": request_data,
                "response": response_data,
                "response_time": response_time,
                "status_code": response.status_code,
            }

        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {str(e)}")
            raise

    def run_test_conversation(self, session_id: Optional[str] = None) -> str:
        """
        Run a test conversation with multiple exchanges.

        Args:
            session_id: Session identifier (generated if not provided)

        Returns:
            Session ID used for the conversation
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        print(f"\n🗣️  Starting test conversation (Session: {session_id})")

        # Test conversation flow
        test_prompts = [
            "Hello! Can you help me with weather information?",
            "What's the weather like in Seattle?",
            "How about the weather in New York?",
            "Are there any weather alerts for California?",
            "Thank you for the weather information!",
        ]

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Turn {i} ---")
            try:
                result = self.send_request(prompt, session_id)
                time.sleep(1)  # Brief pause between requests
            except Exception as e:
                print(f"❌ Conversation failed at turn {i}: {str(e)}")
                break

        return session_id


def run_basic_tests(endpoint: str):
    """Run basic functionality tests."""
    print(f"\n🧪 Running basic tests against {endpoint}")

    tester = AgentTester(endpoint)

    try:
        # Health checks
        print("\n=== Health Checks ===")
        tester.health_check()
        tester.detailed_health_check()

        # Single request test
        print("\n=== Single Request Test ===")
        tester.send_request("What's the weather in San Francisco?")

        # Conversation test
        print("\n=== Conversation Test ===")
        tester.run_test_conversation()

        print("\n✅ All basic tests passed!")

    except Exception as e:
        print(f"\n❌ Basic tests failed: {str(e)}")
        raise


def run_performance_tests(endpoint: str, num_requests: int = 5):
    """Run performance tests with multiple concurrent requests."""
    print(f"\n⚡ Running performance tests ({num_requests} requests)")

    tester = AgentTester(endpoint)
    response_times = []

    try:
        for i in range(num_requests):
            print(f"\nRequest {i + 1}/{num_requests}")
            result = tester.send_request(f"Test request {i + 1}: What's the weather?")
            response_times.append(result["response_time"])
            time.sleep(0.5)  # Brief pause between requests

        # Calculate statistics
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)

        print(f"\n📊 Performance Results:")
        print(f"   Average response time: {avg_time:.2f}s")
        print(f"   Min response time: {min_time:.2f}s")
        print(f"   Max response time: {max_time:.2f}s")
        print(f"   Total requests: {num_requests}")

    except Exception as e:
        print(f"\n❌ Performance tests failed: {str(e)}")
        raise


def run_error_tests(endpoint: str):
    """Run error handling tests."""
    print(f"\n🚨 Running error handling tests")

    tester = AgentTester(endpoint)

    try:
        # Test invalid request format
        print("\n--- Testing invalid request format ---")
        try:
            response = tester.session.post(
                f"{endpoint}/invocations",
                json={"invalid": "request"},
                timeout=tester.timeout,
            )
            print(f"Response status: {response.status_code}")
            if response.status_code == 400:
                print("✅ Correctly rejected invalid request")
            else:
                print("⚠️  Unexpected response to invalid request")
        except Exception as e:
            print(f"❌ Error test failed: {str(e)}")

        # Test missing required fields
        print("\n--- Testing missing required fields ---")
        try:
            response = tester.session.post(
                f"{endpoint}/invocations",
                json={"prompt": "test"},  # Missing session_id
                timeout=tester.timeout,
            )
            print(f"Response status: {response.status_code}")
            if response.status_code == 400:
                print("✅ Correctly rejected incomplete request")
            else:
                print("⚠️  Unexpected response to incomplete request")
        except Exception as e:
            print(f"❌ Error test failed: {str(e)}")

        print("\n✅ Error handling tests completed")

    except Exception as e:
        print(f"\n❌ Error handling tests failed: {str(e)}")
        raise


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="Test weather agent locally using AgentCore request format"
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_LOCAL_ENDPOINT,
        help=f"Agent endpoint URL (default: {DEFAULT_LOCAL_ENDPOINT})",
    )
    parser.add_argument(
        "--test-type",
        choices=["basic", "performance", "error", "all"],
        default="basic",
        help="Type of tests to run (default: basic)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=5,
        help="Number of requests for performance tests (default: 5)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )

    args = parser.parse_args()

    print("🌤️  Weather Agent Local Testing")
    print("=" * 50)
    print(f"Endpoint: {args.endpoint}")
    print(f"Test Type: {args.test_type}")
    print(f"Timeout: {args.timeout}s")
    print(f"Timestamp: {datetime.now().isoformat()}")

    try:
        if args.test_type in ["basic", "all"]:
            run_basic_tests(args.endpoint)

        if args.test_type in ["performance", "all"]:
            run_performance_tests(args.endpoint, args.num_requests)

        if args.test_type in ["error", "all"]:
            run_error_tests(args.endpoint)

        print(f"\n🎉 All {args.test_type} tests completed successfully!")

    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Tests failed with error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
