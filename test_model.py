# test_model.py
import sys
print("Python path:", sys.executable)
print("Current directory:", __import__('os').getcwd())

try:
    from utils.model_manager import get_model_manager
    print("✓ Successfully imported get_model_manager")
    
    # Test creating an instance
    mm = get_model_manager()
    print("✓ Created model manager instance")
    print(f"API Key present: {mm.api_key is not None}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()