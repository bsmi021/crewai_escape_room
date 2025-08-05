#!/usr/bin/env python3
"""
Fix MesaAction constructor calls to include prerequisites parameter
"""

import re
import os

def fix_mesa_action_constructors(file_path):
    """Fix MesaAction constructor calls in a file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern for MesaAction constructor with 4 parameters (missing prerequisites)
    pattern4 = r'MesaAction\(\s*([^,]+),\s*([^,]+),\s*({[^}]*}),\s*([^,\)]+)\s*\)'
    replacement4 = r'MesaAction(\1, \2, \3, \4, [])'
    
    content = re.sub(pattern4, replacement4, content)
    
    # Pattern for MesaAction constructor with named parameters (more complex)
    pattern_named = r'MesaAction\(\s*agent_id=([^,]+),\s*action_type=([^,]+),\s*parameters=({[^}]*}),\s*expected_duration=([^,\)]+)\s*\)'
    replacement_named = r'MesaAction(agent_id=\1, action_type=\2, parameters=\3, expected_duration=\4, prerequisites=[])'
    
    content = re.sub(pattern_named, replacement_named, content)
    
    # Write back if changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed {file_path}")
        return True
    return False

def main():
    test_files = [
        "tests/end_to_end/test_complete_pipeline.py",
        "tests/integration/test_state_unified_management.py", 
        "tests/unit/test_actions_execution.py",
        "tests/unit/test_actions_translation.py",
        "tests/unit/test_hybrid_core_data_models.py"
    ]
    
    fixed_count = 0
    for file_path in test_files:
        if os.path.exists(file_path):
            if fix_mesa_action_constructors(file_path):
                fixed_count += 1
        else:
            print(f"File not found: {file_path}")
    
    print(f"Fixed {fixed_count} files")

if __name__ == "__main__":
    main()