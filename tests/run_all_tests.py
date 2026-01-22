#!/usr/bin/env python3
"""
Master Test Runner for Bitcoin Trading Bot
Runs all test suites and generates comprehensive report
"""

import sys
import os
import subprocess
import time
from datetime import datetime


class TestRunner:
    def __init__(self):
        self.results = {
            'unit_tests': None,
            'integration_tests': None,
            'code_quality': None
        }
        self.start_time = time.time()
    
    def run_all_tests(self):
        """Run all test suites"""
        print("="*70)
        print("🧪 BITCOIN TRADING BOT - COMPREHENSIVE TEST SUITE")
        print("="*70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
        
        # 1. Unit Tests
        print("\n" + "="*70)
        print("1️⃣  RUNNING UNIT TESTS")
        print("="*70 + "\n")
        self.results['unit_tests'] = self.run_unit_tests()
        
        # 2. Integration Tests
        print("\n" + "="*70)
        print("2️⃣  RUNNING INTEGRATION TESTS")
        print("="*70 + "\n")
        self.results['integration_tests'] = self.run_integration_tests()
        
        # 3. Code Quality Analysis
        print("\n" + "="*70)
        print("3️⃣  RUNNING CODE QUALITY ANALYSIS")
        print("="*70 + "\n")
        self.results['code_quality'] = self.run_code_quality()
        
        # Generate final report
        self.generate_final_report()
    
    def run_unit_tests(self):
        """Run unit test suite"""
        try:
            import test_suite
            result = test_suite.run_test_suite()
            return {
                'success': result,
                'error': None
            }
        except Exception as e:
            print(f"❌ Unit tests failed to run: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_integration_tests(self):
        """Run integration test suite"""
        try:
            import test_integration
            result = test_integration.run_integration_tests()
            return {
                'success': result,
                'error': None
            }
        except Exception as e:
            print(f"❌ Integration tests failed to run: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_code_quality(self):
        """Run code quality analysis"""
        try:
            import code_quality_analyzer
            analyzer = code_quality_analyzer.CodeQualityAnalyzer()
            analyzer.analyze_all()
            
            total_issues = sum(len(issues) for issues in analyzer.issues.values())
            critical_issues = sum(
                1 for category in analyzer.issues.values()
                for issue in category
                if issue.get('severity') == 'high'
            )
            
            return {
                'success': critical_issues == 0,
                'total_issues': total_issues,
                'critical_issues': critical_issues,
                'error': None
            }
        except Exception as e:
            print(f"❌ Code quality analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_issues': 0,
                'critical_issues': 0
            }
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        elapsed_time = time.time() - self.start_time
        
        print("\n\n")
        print("="*70)
        print("📊 FINAL TEST REPORT")
        print("="*70)
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {elapsed_time:.2f} seconds")
        print("="*70 + "\n")
        
        # Results summary
        print("TEST RESULTS:")
        print("-" * 70)
        
        all_passed = True
        
        # Unit tests
        unit_result = self.results['unit_tests']
        if unit_result:
            status = "✅ PASSED" if unit_result['success'] else "❌ FAILED"
            print(f"  Unit Tests:        {status}")
            if unit_result['error']:
                print(f"    Error: {unit_result['error']}")
            if not unit_result['success']:
                all_passed = False
        
        # Integration tests
        int_result = self.results['integration_tests']
        if int_result:
            status = "✅ PASSED" if int_result['success'] else "❌ FAILED"
            print(f"  Integration Tests: {status}")
            if int_result['error']:
                print(f"    Error: {int_result['error']}")
            if not int_result['success']:
                all_passed = False
        
        # Code quality
        quality_result = self.results['code_quality']
        if quality_result:
            if quality_result['critical_issues'] == 0:
                status = "✅ PASSED"
            else:
                status = f"⚠️  {quality_result['critical_issues']} critical issues"
                all_passed = False
            
            print(f"  Code Quality:      {status}")
            print(f"    Total issues: {quality_result.get('total_issues', 0)}")
            print(f"    Critical: {quality_result.get('critical_issues', 0)}")
        
        print("\n" + "="*70)
        
        # Overall status
        if all_passed:
            print("🎉 ALL TESTS PASSED!")
            print("="*70)
            return 0
        else:
            print("⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
            print("="*70)
            return 1
    
    def save_report(self, filename="test_report.txt"):
        """Save test report to file"""
        with open(filename, 'w') as f:
            f.write("COMPREHENSIVE TEST REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            for test_type, result in self.results.items():
                f.write(f"\n{test_type.upper()}\n")
                f.write("-"*70 + "\n")
                f.write(f"{result}\n")
        
        print(f"\n💾 Full report saved to {filename}")


def main():
    """Main entry point"""
    runner = TestRunner()
    exit_code = runner.run_all_tests()
    runner.save_report()
    
    print("\n" + "="*70)
    print("📚 NEXT STEPS:")
    print("="*70)
    print("1. Review any failed tests above")
    print("2. Check 'test_report.txt' for detailed results")
    print("3. Check 'code_quality_report.txt' for code issues")
    print("4. Fix critical issues before deploying to production")
    print("="*70 + "\n")
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
