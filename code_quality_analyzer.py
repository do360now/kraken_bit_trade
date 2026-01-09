#!/usr/bin/env python3
"""
Code Quality Analyzer for Bitcoin Trading Bot
Finds duplicate code, code smells, unused code, and potential bugs
"""

import os
import re
import ast
from collections import defaultdict
from typing import List, Dict, Tuple, Set

class CodeQualityAnalyzer:
    def __init__(self, directory="."):
        self.directory = directory
        self.python_files = self._get_python_files()
        self.issues = {
            'duplicate_code': [],
            'code_smells': [],
            'unused_code': [],
            'potential_bugs': [],
            'complexity_issues': []
        }
    
    def _get_python_files(self) -> List[str]:
        """Get all Python files in directory"""
        py_files = []
        for root, dirs, files in os.walk(self.directory):
            # Skip test files and __pycache__
            if '__pycache__' in root or 'test' in root:
                continue
            for file in files:
                if file.endswith('.py') and not file.startswith('test_'):
                    py_files.append(os.path.join(root, file))
        return py_files
    
    def analyze_all(self):
        """Run all analyses"""
        print("🔍 Starting Code Quality Analysis...")
        print(f"   Analyzing {len(self.python_files)} Python files\n")
        
        self.find_duplicate_code()
        self.find_code_smells()
        self.find_unused_code()
        self.find_potential_bugs()
        self.analyze_complexity()
        
        self.generate_report()
    
    def find_duplicate_code(self):
        """Find duplicate code blocks"""
        print("📋 Checking for duplicate code...")
        
        code_blocks = defaultdict(list)
        
        for filepath in self.python_files:
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                
                # Check for duplicate function implementations
                for i in range(len(lines) - 5):
                    block = ''.join(lines[i:i+5]).strip()
                    if block and not block.startswith('#') and len(block) > 50:
                        code_blocks[block].append((filepath, i+1))
            except Exception as e:
                print(f"   ⚠️  Error reading {filepath}: {e}")
        
        # Report duplicates
        for block, locations in code_blocks.items():
            if len(locations) > 1:
                self.issues['duplicate_code'].append({
                    'block': block[:100] + '...',
                    'locations': locations,
                    'severity': 'medium'
                })
    
    def find_code_smells(self):
        """Find common code smells"""
        print("👃 Sniffing for code smells...")
        
        patterns = {
            'long_function': (r'def \w+\([^)]*\):', 100),
            'too_many_parameters': (r'def \w+\(([^)]+)\):', 5),
            'hardcoded_values': (r'= \d{4,}(?!\w)', None),  # Numbers with 4+ digits
            'bare_except': (r'except\s*:', None),
            'mutable_default': (r'def \w+\([^)]*=\s*\[|\{', None),
            'print_statement': (r'\bprint\(', None),  # print() in production code
            'commented_code': (r'^\s*#.*(?:def |class |import |=)', None)
        }
        
        for filepath in self.python_files:
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                
                current_function = None
                function_start = 0
                function_length = 0
                
                for line_num, line in enumerate(lines, 1):
                    # Track function length
                    if line.strip().startswith('def '):
                        if current_function and function_length > 50:
                            self.issues['code_smells'].append({
                                'type': 'long_function',
                                'location': f"{filepath}:{function_start}",
                                'message': f"Function '{current_function}' is {function_length} lines long",
                                'severity': 'medium' if function_length > 100 else 'low'
                            })
                        current_function = line.strip()
                        function_start = line_num
                        function_length = 0
                    
                    if current_function:
                        function_length += 1
                    
                    # Check for bare except
                    if re.search(r'except\s*:', line):
                        self.issues['code_smells'].append({
                            'type': 'bare_except',
                            'location': f"{filepath}:{line_num}",
                            'message': "Bare except clause catches all exceptions",
                            'severity': 'high'
                        })
                    
                    # Check for too many parameters
                    if 'def ' in line:
                        params = re.findall(r'def \w+\(([^)]+)\):', line)
                        if params and params[0].count(',') >= 5:
                            self.issues['code_smells'].append({
                                'type': 'too_many_parameters',
                                'location': f"{filepath}:{line_num}",
                                'message': f"Function has {params[0].count(',') + 1} parameters",
                                'severity': 'medium'
                            })
                    
                    # Check for mutable defaults
                    if 'def ' in line and ('=[]' in line or '={}' in line):
                        self.issues['code_smells'].append({
                            'type': 'mutable_default',
                            'location': f"{filepath}:{line_num}",
                            'message': "Mutable default argument (list/dict)",
                            'severity': 'high'
                        })
                    
                    # Check for hardcoded values (but exclude test files)
                    if not 'test' in filepath:
                        large_numbers = re.findall(r'=\s*(\d{5,})', line)
                        if large_numbers:
                            for num in large_numbers:
                                self.issues['code_smells'].append({
                                    'type': 'magic_number',
                                    'location': f"{filepath}:{line_num}",
                                    'message': f"Magic number: {num}",
                                    'severity': 'low'
                                })
                
            except Exception as e:
                print(f"   ⚠️  Error analyzing {filepath}: {e}")
    
    def find_unused_code(self):
        """Find potentially unused code"""
        print("🗑️  Looking for unused code...")
        
        defined_functions = set()
        called_functions = set()
        imported_modules = defaultdict(set)
        used_modules = defaultdict(set)
        
        for filepath in self.python_files:
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                
                # Find function definitions
                func_defs = re.findall(r'def (\w+)\(', content)
                defined_functions.update(func_defs)
                
                # Find function calls
                func_calls = re.findall(r'(\w+)\(', content)
                called_functions.update(func_calls)
                
                # Find imports
                imports = re.findall(r'import (\w+)', content)
                imports.extend(re.findall(r'from (\w+) import', content))
                imported_modules[filepath].update(imports)
                
                # Track module usage
                for module in imports:
                    if module in content:
                        used_modules[filepath].add(module)
                
            except Exception as e:
                print(f"   ⚠️  Error analyzing {filepath}: {e}")
        
        # Report unused functions
        unused_functions = defined_functions - called_functions
        for func in unused_functions:
            if not func.startswith('_'):  # Ignore private functions
                self.issues['unused_code'].append({
                    'type': 'unused_function',
                    'name': func,
                    'severity': 'low'
                })
        
        # Report unused imports
        for filepath, imports in imported_modules.items():
            unused_imports = imports - used_modules[filepath]
            for module in unused_imports:
                self.issues['unused_code'].append({
                    'type': 'unused_import',
                    'location': filepath,
                    'name': module,
                    'severity': 'low'
                })
    
    def find_potential_bugs(self):
        """Find potential bugs"""
        print("🐛 Hunting for potential bugs...")
        
        bug_patterns = [
            (r'if.*=(?!=)', 'Assignment in condition'),
            (r'==\s*None', 'Use "is None" instead of "== None"'),
            (r'except\s+\w+\s*,\s*\w+:', 'Old-style except syntax'),
            (r'\.get\(\w+\)\s*\[', 'Unsafe chained access on .get()'),
            (r'/\s*0(?!\w)', 'Division by zero'),
            (r'float\(["\']nan["\']', 'Creating NaN float'),
            (r'return.*and.*or', 'Complex return with and/or (use if/else)'),
        ]
        
        for filepath in self.python_files:
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    for pattern, message in bug_patterns:
                        if re.search(pattern, line) and not line.strip().startswith('#'):
                            self.issues['potential_bugs'].append({
                                'location': f"{filepath}:{line_num}",
                                'message': message,
                                'code': line.strip(),
                                'severity': 'high'
                            })
            except Exception as e:
                print(f"   ⚠️  Error analyzing {filepath}: {e}")
    
    def analyze_complexity(self):
        """Analyze code complexity"""
        print("📊 Analyzing code complexity...")
        
        for filepath in self.python_files:
            try:
                with open(filepath, 'r') as f:
                    tree = ast.parse(f.read(), filename=filepath)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Count branches (if, for, while, etc.)
                        branches = sum(1 for _ in ast.walk(node) if isinstance(_, (ast.If, ast.For, ast.While, ast.ExceptHandler)))
                        
                        if branches > 10:
                            self.issues['complexity_issues'].append({
                                'type': 'high_complexity',
                                'location': f"{filepath}:{node.lineno}",
                                'function': node.name,
                                'branches': branches,
                                'severity': 'high' if branches > 15 else 'medium'
                            })
                        
                        # Check nesting depth
                        max_depth = self._calculate_nesting_depth(node)
                        if max_depth > 4:
                            self.issues['complexity_issues'].append({
                                'type': 'deep_nesting',
                                'location': f"{filepath}:{node.lineno}",
                                'function': node.name,
                                'depth': max_depth,
                                'severity': 'medium'
                            })
            
            except Exception as e:
                print(f"   ⚠️  Error analyzing {filepath}: {e}")
    
    def _calculate_nesting_depth(self, node, depth=0):
        """Calculate maximum nesting depth"""
        max_depth = depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With)):
                child_depth = self._calculate_nesting_depth(child, depth + 1)
                max_depth = max(max_depth, child_depth)
        return max_depth
    
    def generate_report(self):
        """Generate comprehensive report"""
        print("\n" + "="*70)
        print("CODE QUALITY REPORT")
        print("="*70 + "\n")
        
        total_issues = sum(len(issues) for issues in self.issues.values())
        
        print(f"📊 Total Issues Found: {total_issues}\n")
        
        # Summary by category
        for category, issues in self.issues.items():
            if issues:
                print(f"{'='*70}")
                print(f"{category.replace('_', ' ').upper()}: {len(issues)} issues")
                print(f"{'='*70}")
                
                # Group by severity
                high = [i for i in issues if i.get('severity') == 'high']
                medium = [i for i in issues if i.get('severity') == 'medium']
                low = [i for i in issues if i.get('severity') == 'low']
                
                if high:
                    print(f"\n🔴 HIGH SEVERITY ({len(high)}):")
                    for issue in high[:5]:  # Show first 5
                        self._print_issue(issue)
                
                if medium:
                    print(f"\n🟡 MEDIUM SEVERITY ({len(medium)}):")
                    for issue in medium[:5]:
                        self._print_issue(issue)
                
                if low:
                    print(f"\n🟢 LOW SEVERITY ({len(low)}):")
                    for issue in low[:3]:
                        self._print_issue(issue)
                
                print()
        
        # Priority recommendations
        print("="*70)
        print("🎯 PRIORITY RECOMMENDATIONS")
        print("="*70)
        
        recommendations = []
        
        # Bare excepts are dangerous
        bare_excepts = [i for i in self.issues['code_smells'] if i['type'] == 'bare_except']
        if bare_excepts:
            recommendations.append(f"1. Fix {len(bare_excepts)} bare except clauses (can hide bugs)")
        
        # Mutable defaults are bugs waiting to happen
        mutable_defaults = [i for i in self.issues['code_smells'] if i['type'] == 'mutable_default']
        if mutable_defaults:
            recommendations.append(f"2. Fix {len(mutable_defaults)} mutable default arguments (common bug source)")
        
        # High complexity functions
        high_complexity = [i for i in self.issues['complexity_issues'] if i.get('branches', 0) > 15]
        if high_complexity:
            recommendations.append(f"3. Refactor {len(high_complexity)} highly complex functions")
        
        # Potential bugs
        if self.issues['potential_bugs']:
            recommendations.append(f"4. Review {len(self.issues['potential_bugs'])} potential bugs")
        
        # Duplicate code
        if len(self.issues['duplicate_code']) > 3:
            recommendations.append(f"5. Refactor duplicate code blocks into reusable functions")
        
        for rec in recommendations:
            print(f"   {rec}")
        
        print("\n" + "="*70 + "\n")
    
    def _print_issue(self, issue):
        """Print a single issue"""
        if 'location' in issue:
            print(f"   📍 {issue['location']}")
        if 'message' in issue:
            print(f"      {issue['message']}")
        if 'code' in issue:
            print(f"      Code: {issue['code'][:60]}...")
        if 'function' in issue:
            print(f"      Function: {issue['function']}")
        print()
    
    def save_report(self, filename="code_quality_report.txt"):
        """Save report to file"""
        with open(filename, 'w') as f:
            f.write("CODE QUALITY ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            
            for category, issues in self.issues.items():
                f.write(f"\n{category.upper()}\n")
                f.write(f"{'-'*70}\n")
                for issue in issues:
                    f.write(f"{issue}\n")
        
        print(f"💾 Report saved to {filename}")


def main():
    # Analyze code in current directory
    analyzer = CodeQualityAnalyzer()
    analyzer.analyze_all()
    analyzer.save_report()


if __name__ == "__main__":
    main()
