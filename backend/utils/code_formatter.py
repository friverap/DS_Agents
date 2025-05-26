"""
Code formatting utilities for DSAgency
Based on Auto-Analyst best practices for Python code formatting
"""

import re
import textwrap


def clean_code_block(code_str: str) -> str:
    """
    Clean and format a Python code block for better readability and execution.
    
    Args:
        code_str: Raw code string that may contain markdown formatting
        
    Returns:
        Cleaned and formatted code string
    """
    # Remove markdown code block markers
    code_clean = re.sub(r'^```python\n?', '', code_str, flags=re.MULTILINE)
    code_clean = re.sub(r'\n```$', '', code_clean)
    code_clean = re.sub(r'^```\n?', '', code_clean, flags=re.MULTILINE)
    
    # Remove any trailing markdown markers
    code_clean = code_clean.strip()
    if code_clean.endswith('```'):
        code_clean = code_clean[:-3].strip()
    
    return code_clean


def format_code_with_proper_lines(code_str: str) -> str:
    """
    Format code to ensure proper line breaks and structure.
    
    Args:
        code_str: Code string that may have formatting issues
        
    Returns:
        Properly formatted code string
    """
    # Clean the code block first
    code_clean = clean_code_block(code_str)
    
    # Handle the case where code is all on one line or has missing line breaks
    # Look for common Python patterns and add line breaks
    
    # First, handle import statements that might be concatenated
    code_clean = re.sub(r'import\s+(\w+)import\s+', r'import \1\nimport ', code_clean)
    code_clean = re.sub(r'import\s+([^#\n]+?)([A-Z]\w+\s*=)', r'import \1\n\2', code_clean)
    
    # Handle comments that are stuck to code
    code_clean = re.sub(r'([^#\n]+)(#[^#\n]*?)([A-Z]\w+)', r'\1\2\n\3', code_clean)
    
    # Add line breaks after common Python statements
    patterns_for_newlines = [
        (r'(\w+\s*=\s*[^#\n]+?)([A-Z]\w+\s*=)', r'\1\n\2'),  # Variable assignments
        (r'(plt\.[^#\n]+?)([A-Z]\w+)', r'\1\n\2'),  # Matplotlib calls
        (r'(print\([^)]+\))([A-Z]\w+)', r'\1\n\2'),  # Print statements
        (r'(plt\.show\(\))([A-Z]\w+)', r'\1\n\2'),  # plt.show()
        (r'(plt\.figure\([^)]*\))([A-Z]\w+)', r'\1\n\2'),  # plt.figure()
        (r'(\))([A-Z]\w+\s*=)', r')\n\2'),  # Function calls followed by assignments
        (r'(#[^#\n]*?)([A-Z]\w+)', r'\1\n\2'),  # Comments followed by code
        (r'(\w+\s*=\s*\[[^\]]+\])([A-Z]\w+)', r'\1\n\2'),  # List assignments
        (r'(plt\.[^(]+\([^)]*\))([A-Z]\w+)', r'\1\n\2'),  # Matplotlib function calls
    ]
    
    for pattern, replacement in patterns_for_newlines:
        code_clean = re.sub(pattern, replacement, code_clean)
    
    # Split into lines and process each line
    lines = code_clean.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            formatted_lines.append('')
            continue
            
        # Handle common cases where multiple statements might be on one line
        # Split on semicolons (but be careful with strings)
        if ';' in line and not _is_in_string(line, line.find(';')):
            parts = line.split(';')
            for part in parts:
                part = part.strip()
                if part:
                    formatted_lines.append(part)
        else:
            # Check if this line has multiple statements without semicolons
            # Look for patterns like: variable = valueOtherVariable = otherValue
            if re.search(r'\w+\s*=\s*[^=]+[A-Z]\w+\s*=', line):
                # Try to split on assignment patterns
                parts = re.split(r'(?<=[^=])(?=[A-Z]\w+\s*=)', line)
                for part in parts:
                    part = part.strip()
                    if part:
                        formatted_lines.append(part)
            else:
                formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)


def _is_in_string(text: str, position: int) -> bool:
    """
    Check if a position in text is inside a string literal.
    
    Args:
        text: The text to check
        position: The position to check
        
    Returns:
        True if position is inside a string literal
    """
    # Simple check for string literals
    before_pos = text[:position]
    single_quotes = before_pos.count("'") - before_pos.count("\\'")
    double_quotes = before_pos.count('"') - before_pos.count('\\"')
    
    # If odd number of quotes, we're inside a string
    return (single_quotes % 2 == 1) or (double_quotes % 2 == 1)


def add_proper_spacing(code_str: str) -> str:
    """
    Add proper spacing around operators and after commas.
    
    Args:
        code_str: Code string to format
        
    Returns:
        Code string with proper spacing
    """
    # Add spaces around operators (but be careful with strings)
    patterns = [
        (r'([^=!<>])=([^=])', r'\1 = \2'),  # Assignment
        (r'([^=!<>])==([^=])', r'\1 == \2'),  # Equality
        (r'([^=!<>])!=([^=])', r'\1 != \2'),  # Not equal
        (r'([^<])<=([^=])', r'\1 <= \2'),  # Less than or equal
        (r'([^>])>=([^=])', r'\1 >= \2'),  # Greater than or equal
        (r'([^<])<([^=])', r'\1 < \2'),   # Less than
        (r'([^>])>([^=])', r'\1 > \2'),   # Greater than
        (r'([^+])\+([^=+])', r'\1 + \2'), # Addition
        (r'([^-])-([^=\->])', r'\1 - \2'), # Subtraction
        (r'([^*])\*([^=*])', r'\1 * \2'), # Multiplication
        (r'([^/])/([^=])', r'\1 / \2'),   # Division
        (r',([^\s])', r', \1'),           # Comma spacing
    ]
    
    formatted_code = code_str
    for pattern, replacement in patterns:
        formatted_code = re.sub(pattern, replacement, formatted_code)
    
    return formatted_code


def clean_imports(code_str: str) -> str:
    """
    Clean and organize import statements.
    
    Args:
        code_str: Code string with imports
        
    Returns:
        Code string with cleaned imports
    """
    lines = code_str.split('\n')
    import_lines = []
    other_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            import_lines.append(line)
        else:
            other_lines.append(line)
    
    # Remove duplicate imports
    unique_imports = []
    seen_imports = set()
    
    for imp in import_lines:
        if imp.strip() not in seen_imports:
            unique_imports.append(imp)
            seen_imports.add(imp.strip())
    
    # Combine imports and other code
    if unique_imports and other_lines:
        # Add a blank line after imports if there's other code
        result = unique_imports + [''] + other_lines
    else:
        result = unique_imports + other_lines
    
    return '\n'.join(result)


def format_python_code(code_str: str) -> str:
    """
    Main function to format Python code for better readability and execution.
    
    Args:
        code_str: Raw Python code string
        
    Returns:
        Formatted Python code string
    """
    # Step 1: Clean code block markers
    code = clean_code_block(code_str)
    
    # Step 2: Format with proper line breaks
    code = format_code_with_proper_lines(code)
    
    # Step 3: Add proper spacing
    code = add_proper_spacing(code)
    
    # Step 4: Clean imports
    code = clean_imports(code)
    
    # Step 5: Remove excessive blank lines
    code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)
    
    # Step 6: Ensure proper indentation (basic)
    lines = code.split('\n')
    formatted_lines = []
    indent_level = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            formatted_lines.append('')
            continue
            
        # Decrease indent for certain keywords
        if stripped.startswith(('except', 'elif', 'else', 'finally')):
            indent_level = max(0, indent_level - 1)
        elif stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'with ')):
            pass  # Keep current indent
        elif stripped.endswith(':'):
            pass  # Keep current indent, will increase after
        
        # Apply current indentation
        formatted_lines.append('    ' * indent_level + stripped)
        
        # Increase indent after certain patterns
        if stripped.endswith(':') and not stripped.startswith('#'):
            indent_level += 1
    
    return '\n'.join(formatted_lines).strip()


def create_executable_code_block(code_str: str) -> str:
    """
    Create a properly formatted code block ready for execution.
    
    Args:
        code_str: Raw code string
        
    Returns:
        Formatted code block with markdown markers
    """
    formatted_code = format_python_code(code_str)
    return f"```python\n{formatted_code}\n```"


# Example usage and testing
if __name__ == "__main__":
    # Test with problematic code
    test_code = """
import matplotlib.pyplot as pltimport numpy as npx = np.linspace(0, 10, 100)y = np.sin(x)plt.figure(figsize=(10, 6))plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')plt.title('Sine Function')plt.xlabel('X')plt.ylabel('Y')plt.grid(True)plt.legend()plt.show()print("Plot generated!")
    """
    
    print("Original code:")
    print(repr(test_code))
    print("\nFormatted code:")
    formatted = format_python_code(test_code)
    print(formatted)
    print("\nExecutable block:")
    print(create_executable_code_block(test_code)) 