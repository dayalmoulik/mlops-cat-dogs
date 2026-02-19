# Smoke Test Script for Deployment

import requests
import time
import sys

def smoke_test(api_url):
    """Run smoke tests on deployed API"""
    
    print("=" * 60)
    print("Running Smoke Tests")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Health Check
    print("\n1. Testing Health Endpoint...")
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy":
                print("   ✓ Health check passed")
                tests_passed += 1
            else:
                print(f"   ✗ Health check failed: {data}")
                tests_failed += 1
        else:
            print(f"   ✗ Health check failed: Status {response.status_code}")
            tests_failed += 1
    except Exception as e:
        print(f"   ✗ Health check error: {e}")
        tests_failed += 1
    
    # Test 2: Root Endpoint
    print("\n2. Testing Root Endpoint...")
    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "version" in data:
                print(f"   ✓ Root endpoint passed (version: {data['version']})")
                tests_passed += 1
            else:
                print("   ✗ Root endpoint missing version")
                tests_failed += 1
        else:
            print(f"   ✗ Root endpoint failed: Status {response.status_code}")
            tests_failed += 1
    except Exception as e:
        print(f"   ✗ Root endpoint error: {e}")
        tests_failed += 1
    
    # Test 3: Model Info
    print("\n3. Testing Model Info Endpoint...")
    try:
        response = requests.get(f"{api_url}/model/info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "model_type" in data:
                print(f"   ✓ Model info passed (type: {data['model_type']})")
                tests_passed += 1
            else:
                print("   ✗ Model info incomplete")
                tests_failed += 1
        else:
            print(f"   ✗ Model info failed: Status {response.status_code}")
            tests_failed += 1
    except Exception as e:
        print(f"   ✗ Model info error: {e}")
        tests_failed += 1
    
    # Test 4: Response Time
    print("\n4. Testing Response Time...")
    try:
        start = time.time()
        response = requests.get(f"{api_url}/health", timeout=10)
        end = time.time()
        response_time = (end - start) * 1000
        
        if response_time < 1000:  # Less than 1 second
            print(f"   ✓ Response time acceptable: {response_time:.2f}ms")
            tests_passed += 1
        else:
            print(f"   ⚠ Response time slow: {response_time:.2f}ms")
            tests_passed += 1  # Still pass but warn
    except Exception as e:
        print(f"   ✗ Response time test error: {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("Smoke Test Results")
    print("=" * 60)
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")
    print("=" * 60)
    
    if tests_failed == 0:
        print("✓ All smoke tests passed!")
        return 0
    else:
        print("✗ Some smoke tests failed!")
        return 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
    else:
        api_url = "http://localhost:8000"
    
    print(f"Testing API at: {api_url}")
    exit_code = smoke_test(api_url)
    sys.exit(exit_code)
