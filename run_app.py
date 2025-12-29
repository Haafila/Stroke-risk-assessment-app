#!/usr/bin/env python3
"""
Script to run the Stroke Risk Assessment app
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    required_packages = {
        'streamlit': 'streamlit',
        'pandas': 'pandas', 
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'joblib': 'joblib',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing_packages = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install requirements:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def check_model_files():
    """Check if model files exist"""
    model_files = [
        'stroke_prediction_pipeline.pkl',
        'model_metadata.pkl'
    ]
    
    missing_files = []
    for file in model_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing model files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n🔧 Generate model files:")
        print("   1. Run the complete Jupyter notebook (cerebral-stroke-combined.ipynb)")
        print("   2. Ensure model saving cells are executed")
        print("   3. Make sure these files are created in the same directory")
        return False
    
    return True

def main():
    """Main function to run the app"""
    print("🏥 Stroke Risk Assessment - Complete Package")
    print("=" * 50)
    
    # Check requirements
    print("📦 Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    print("✅ All requirements satisfied")
    
    # Check model files
    print("🔍 Checking model files...")
    if not check_model_files():
        sys.exit(1)
    print("✅ Model files found")
    
    # Run Streamlit app
    print("🚀 Starting Streamlit app...")
    print("📱 Open your browser to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

if __name__ == "__main__":
    main()

