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
Simple test script for the Weather Tools MCP Server.

This script demonstrates how to interact with the MCP server by making
HTTP requests to test the available tools.
"""

import json
import requests
import time


def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check...")
    try:
        response = requests.get("http://localhost:8000/health")
        response.raise_for_status()
        print(f"✓ Health check passed: {response.json()}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_list_tools():
    """Test listing available tools."""
    print("\nTesting list tools...")
    try:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}

        response = requests.post(
            "http://localhost:8000/mcp",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        result = response.json()
        if "result" in result and "tools" in result["result"]:
            tools = result["result"]["tools"]
            print(f"✓ Found {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool['name']}: {tool['description'][:100]}...")
            return True
        else:
            print(f"✗ Unexpected response format: {result}")
            return False

    except Exception as e:
        print(f"✗ List tools failed: {e}")
        return False


def test_weather_forecast():
    """Test the weather forecast tool."""
    print("\nTesting weather forecast tool...")
    try:
        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "get_weather_forecast",
                "arguments": {"location": "Seattle, WA", "days": 3},
            },
        }

        response = requests.post(
            "http://localhost:8000/mcp",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        result = response.json()
        if "result" in result and "content" in result["result"]:
            content = result["result"]["content"][0]["text"]
            forecast_data = json.loads(content)

            if forecast_data.get("success"):
                print(f"✓ Weather forecast for {forecast_data['location']}:")
                for day in forecast_data.get("forecast", []):
                    print(
                        f"  {day['date']}: {day['conditions']}, High: {day['temperature_high']}°F, Low: {day['temperature_low']}°F"
                    )
                return True
            else:
                print(
                    f"✗ Forecast failed: {forecast_data.get('error', 'Unknown error')}"
                )
                return False
        else:
            print(f"✗ Unexpected response format: {result}")
            return False

    except Exception as e:
        print(f"✗ Weather forecast test failed: {e}")
        return False


def test_weather_alerts():
    """Test the weather alerts tool."""
    print("\nTesting weather alerts tool...")
    try:
        payload = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "get_weather_alerts", "arguments": {"area": "CA"}},
        }

        response = requests.post(
            "http://localhost:8000/mcp",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        result = response.json()
        if "result" in result and "content" in result["result"]:
            content = result["result"]["content"][0]["text"]
            alert_data = json.loads(content)

            if alert_data.get("success"):
                alert_count = alert_data.get("alert_count", 0)
                print(
                    f"✓ Weather alerts for {alert_data['area']}: {alert_count} alerts found"
                )
                for alert in alert_data.get("alerts", []):
                    print(
                        f"  - {alert['event']}: {alert['severity']} ({alert['area_desc']})"
                    )
                return True
            else:
                print(f"✗ Alerts failed: {alert_data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"✗ Unexpected response format: {result}")
            return False

    except Exception as e:
        print(f"✗ Weather alerts test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Weather Tools MCP Server Test Suite")
    print("=" * 40)

    # Wait a moment for server to be ready
    print("Waiting for server to be ready...")
    time.sleep(2)

    tests = [
        test_health_check,
        test_list_tools,
        test_weather_forecast,
        test_weather_alerts,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n{'=' * 40}")
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
