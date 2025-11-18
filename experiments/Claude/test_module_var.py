#!/usr/bin/env python3
"""
Test module variable updates from external code.
"""

# Create a mock module
import sys
from types import ModuleType

# Create a test module
test_module = ModuleType('test_module')

# Add module-level variable
test_module.OUTPUT_DIR = "initial"

# Add functions that read the variable
exec("""
def get_path():
    return OUTPUT_DIR + "/file.txt"
""", test_module.__dict__)

# Test 1: Initial value
print("Test 1: Initial value")
print(f"  test_module.OUTPUT_DIR = {test_module.OUTPUT_DIR}")
print(f"  test_module.get_path() = {test_module.get_path()}")
assert test_module.get_path() == "initial/file.txt"
print("  ✓ Pass")

# Test 2: Update from outside
print("\nTest 2: Update from outside")
test_module.OUTPUT_DIR = "updated"
print(f"  test_module.OUTPUT_DIR = {test_module.OUTPUT_DIR}")
print(f"  test_module.get_path() = {test_module.get_path()}")
assert test_module.get_path() == "updated/file.txt"
print("  ✓ Pass")

print("\n✓ Module variable updates work correctly!")
