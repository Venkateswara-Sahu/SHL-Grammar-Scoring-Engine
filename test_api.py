"""Test API endpoints with sample requests."""
import requests
import json
import sys
import time


API_BASE_URL = "http://localhost:8001"


def test_health():
    """Test health endpoint."""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)
    
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_text_scoring():
    """Test text scoring endpoint."""
    print("\n" + "="*60)
    print("Testing Text Scoring Endpoint")
    print("="*60)
    
    data = {
        "text": "Hello, my name is John and I am very excited to be here today.",
        "return_detailed": True
    }
    
    response = requests.post(
        f"{API_BASE_URL}/score/text",
        json=data
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nGrammar Score: {result['grammar_score']}/10.0")
        print(f"Grade: {result['grade']}")
        print(f"Success: {result['success']}")
        if result.get('metrics'):
            print(f"Error Count: {result['metrics']['error_count']}")
        if result.get('detailed_analysis'):
            print(f"Detailed Analysis Available: Yes")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200


def test_audio_scoring(audio_file: str):
    """Test audio scoring endpoint."""
    print("\n" + "="*60)
    print("Testing Audio Scoring Endpoint")
    print("="*60)
    
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            data = {
                'preprocess': True,
                'return_detailed': True
            }
            
            response = requests.post(
                f"{API_BASE_URL}/score/audio",
                files=files,
                data=data
            )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nGrammar Score: {result['grammar_score']}/10.0")
            print(f"Grade: {result['grade']}")
            print(f"Transcription: {result['transcription']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Processing Time: {result['processing_time']}s")
        else:
            print(f"Error: {response.text}")
        
        return response.status_code == 200
        
    except FileNotFoundError:
        print(f"Audio file not found: {audio_file}")
        return False


def test_api_info():
    """Test API info endpoint."""
    print("\n" + "="*60)
    print("Testing API Info Endpoint")
    print("="*60)
    
    response = requests.get(f"{API_BASE_URL}/api/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     Grammar Scoring Engine - API Test Suite             ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    print("\nMake sure the API server is running at http://localhost:8001")
    print("Start server with: python src/api/main.py")
    
    time.sleep(2)
    
    # Run tests
    results = {
        "Health Check": test_health(),
        "API Info": test_api_info(),
        "Text Scoring": test_text_scoring()
    }
    
    # Test audio scoring if file provided
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        results["Audio Scoring"] = test_audio_scoring(audio_file)
    else:
        print("\n" + "="*60)
        print("Skipping audio test (no file provided)")
        print("Usage: python test_api.py <path_to_audio_file>")
        print("="*60)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print("="*60 + "\n")
    
    # Exit code
    sys.exit(0 if all(results.values()) else 1)
