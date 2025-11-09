"""Verification script - checks if project is ready for submission."""
import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists."""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists

def check_directory_exists(dirpath, description):
    """Check if a directory exists."""
    exists = os.path.isdir(dirpath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {dirpath}")
    return exists

def check_import(module_name):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✅ {module_name} is installed")
        return True
    except ImportError:
        print(f"❌ {module_name} is NOT installed")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     Grammar Scoring Engine - Project Verification       ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    all_checks = []
    
    # Check project structure
    print("\n[1/5] Checking Project Structure...")
    print("-" * 60)
    
    structure_checks = [
        check_directory_exists("src", "Source directory"),
        check_directory_exists("src/audio", "Audio module"),
        check_directory_exists("src/asr", "ASR module"),
        check_directory_exists("src/grammar", "Grammar module"),
        check_directory_exists("src/pipeline", "Pipeline module"),
        check_directory_exists("src/api", "API module"),
        check_directory_exists("tests", "Tests directory"),
        check_directory_exists("data", "Data directory"),
    ]
    all_checks.extend(structure_checks)
    
    # Check key files
    print("\n[2/5] Checking Key Files...")
    print("-" * 60)
    
    file_checks = [
        check_file_exists("requirements.txt", "Requirements file"),
        check_file_exists("README.md", "README file"),
        check_file_exists("src/pipeline/grammar_scorer.py", "Main pipeline"),
        check_file_exists("src/api/main.py", "API main file"),
        check_file_exists("example_usage.py", "Example usage"),
        check_file_exists("evaluate_dataset.py", "Evaluation script"),
        check_file_exists("test_api.py", "API test script"),
    ]
    all_checks.extend(file_checks)
    
    # Check dependencies
    print("\n[3/5] Checking Python Dependencies...")
    print("-" * 60)
    
    dependency_checks = [
        check_import("whisper"),
        check_import("torch"),
        check_import("librosa"),
        check_import("spacy"),
        check_import("nltk"),
        check_import("language_tool_python"),
        check_import("fastapi"),
        check_import("pandas"),
    ]
    all_checks.extend(dependency_checks)
    
    # Check spaCy model
    print("\n[4/5] Checking NLP Models...")
    print("-" * 60)
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy English model (en_core_web_sm) is loaded")
        all_checks.append(True)
    except:
        print("❌ spaCy English model (en_core_web_sm) is NOT loaded")
        print("   Run: python -m spacy download en_core_web_sm")
        all_checks.append(False)
    
    # Check FFmpeg
    print("\n[5/5] Checking External Tools...")
    print("-" * 60)
    
    ffmpeg_available = os.system("ffmpeg -version >nul 2>&1") == 0
    if ffmpeg_available:
        print("✅ FFmpeg is installed")
        all_checks.append(True)
    else:
        print("⚠️  FFmpeg is NOT installed (recommended for audio processing)")
        print("   Download from: https://ffmpeg.org/download.html")
        all_checks.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    passed = sum(all_checks)
    total = len(all_checks)
    percentage = (passed / total) * 100
    
    print(f"\nChecks Passed: {passed}/{total} ({percentage:.1f}%)")
    
    if percentage == 100:
        print("\n✅ ✅ ✅ PROJECT IS READY FOR SUBMISSION! ✅ ✅ ✅")
        print("\nNext steps:")
        print("  1. Test: python example_usage.py")
        print("  2. Evaluate dataset: python evaluate_dataset.py")
        print("  3. Create GitHub repo")
        print("  4. Submit via SHL form")
    elif percentage >= 90:
        print("\n⚠️  PROJECT IS MOSTLY READY")
        print("Fix the failed checks above for best results.")
    else:
        print("\n❌ PROJECT NEEDS ATTENTION")
        print("Please fix the failed checks above.")
    
    print("\n" + "="*60 + "\n")
    
    return percentage == 100


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
