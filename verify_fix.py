import pandas as pd
import glob
import os

# --- 1. SETUP: Create a dummy environment ---
print("[ASSISTANT] 🛠️  Setting up test environment...")
dummy_file = "test_verification.parquet"
df = pd.DataFrame({'col1': [2, 3], 'col2': [4, 5]})
df.to_parquet(dummy_file)

# Simulate what 'glob.glob' returns in your system
files = [dummy_file] 
print(f"[ASSISTANT] 📂 'files' variable contains: {files} (Type: {type(files)})")

# --- 2. REPRODUCE THE BUG (What you have now) ---
print("\n[ASSISTANT] 🧪 Testing Current Code (test_file = files)...")
try:
    test_file = files # <--- THE BUG
    pd.read_parquet(test_file)
    print("❌ FAILURE: The buggy code somehow worked? (This shouldn't happen)")
except Exception as e:
    print(f"✅ VERIFIED BUG: Crashed as expected with error: {e}")

# --- 3. VERIFY THE SOLUTION (What I am proposing) ---
print("\n[ASSISTANT] 🧪 Testing Fix (test_file = files)...")
try:
    test_file = files # <--- THE FIX
    pd.read_parquet(test_file)
    print(f"✅ SUCCESS: Successfully read file using files")
except Exception as e:
    print(f"❌ FAILURE: The fix failed: {e}")

# Cleanup
if os.path.exists(dummy_file):
    os.remove(dummy_file)