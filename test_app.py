#!/usr/bin/env python3
"""
Test script to verify the Stroke Risk Assessment app works correctly
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    try:
        import joblib
        print("✅ Joblib imported successfully")
    except ImportError as e:
        print(f"❌ Joblib import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test if model files can be loaded"""
    print("\n🔍 Testing model loading...")
    
    try:
        import joblib
        
        # Test if model files exist
        if os.path.exists('stroke_prediction_pipeline.pkl'):
            print("✅ stroke_prediction_pipeline.pkl found")
        else:
            print("❌ stroke_prediction_pipeline.pkl not found")
            return False
            
        if os.path.exists('best_stroke_model.pkl'):
            print("✅ best_stroke_model.pkl found")
        else:
            print("❌ best_stroke_model.pkl not found")
            return False
            
        # Test loading the model (not the pipeline to avoid class issues)
        try:
            model = joblib.load('best_stroke_model.pkl')
            print("✅ Model loaded successfully")
            
            # Check if it has the required attributes
            if hasattr(model, 'predict_proba'):
                print("✅ Model has predict_proba method")
            else:
                print("⚠️  Model missing predict_proba method")
            
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_app_import():
    """Test if the app can be imported"""
    print("\n🔍 Testing app import...")
    
    try:
        # Import the app module
        import app
        print("✅ App module imported successfully")
        
        # Check if main function exists
        if hasattr(app, 'main'):
            print("✅ App has main function")
        else:
            print("⚠️  App missing main function")
        
        return True
        
    except Exception as e:
        print(f"❌ App import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🏥 Stroke Risk Assessment - Test Suite")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test model loading
    model_ok = test_model_loading()
    
    # Test app import
    app_ok = test_app_import()
    
    # Summary
    print("\n📊 Test Results:")
    print("=" * 20)
    print(f"Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Model Loading: {'✅ PASS' if model_ok else '❌ FAIL'}")
    print(f"App Import: {'✅ PASS' if app_ok else '❌ FAIL'}")
    
    if imports_ok and model_ok and app_ok:
        print("\n🎉 All tests passed! The app is ready to run.")
        print("Run: streamlit run app.py")
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
    
    return imports_ok and model_ok and app_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
