
import test_ablation
import system_config as cfg
from continuous_learning_analyzer import run_simulation
import sys

# Configure for a quick test
test_ablation.START_DATE = "2025-01-01"
test_ablation.END_DATE = "2025-02-01"

print("Running verification test for ablation suite...")
try:
    test_ablation.run_ablation_suite()
    print("\nVerification SUCCESS: Ablation suite ran without errors.")
except Exception as e:
    print(f"\nVerification FAILED: {e}")
    import traceback
    traceback.print_exc()
