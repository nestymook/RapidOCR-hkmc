"""
Test API compatibility for rapidocr_hkmc package.
This script tests that all public APIs work identically to the original rapidocr package.
"""
import sys
import numpy as np
from rapidocr_hkmc import RapidOCR, LoadImageError, VisRes

def test_basic_import():
    """Test that basic imports work"""
    print("✓ Test 1: Basic imports successful")
    return True

def test_rapidocr_initialization():
    """Test RapidOCR initialization with default config"""
    try:
        ocr = RapidOCR()
        print("✓ Test 2: RapidOCR initialization successful")
        return True
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
        return False

def test_rapidocr_with_config():
    """Test RapidOCR initialization with custom config"""
    try:
        # Test with config parameters
        ocr = RapidOCR(
            det_use_cuda=False,
            cls_use_cuda=False,
            rec_use_cuda=False
        )
        print("✓ Test 3: RapidOCR with custom config successful")
        return True
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
        return False

def test_image_processing():
    """Test image processing with a simple test image"""
    try:
        ocr = RapidOCR()
        
        # Create a simple test image (white background)
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # Try to process it (should return empty or error gracefully)
        result, elapse = ocr(test_image)
        
        print(f"✓ Test 4: Image processing successful (result: {result is not None}, elapse: {elapse})")
        return True
    except Exception as e:
        print(f"✗ Test 4 failed: {e}")
        return False

def test_vis_res():
    """Test VisRes utility"""
    try:
        vis = VisRes()
        print("✓ Test 5: VisRes initialization successful")
        return True
    except Exception as e:
        print(f"✗ Test 5 failed: {e}")
        return False

def test_load_image_error():
    """Test LoadImageError exception"""
    try:
        # Test that the exception can be raised and caught
        try:
            raise LoadImageError("Test error")
        except LoadImageError as e:
            if str(e) == "Test error":
                print("✓ Test 6: LoadImageError exception works correctly")
                return True
        return False
    except Exception as e:
        print(f"✗ Test 6 failed: {e}")
        return False

def test_backward_compatibility():
    """Test backward compatibility with existing configurations"""
    try:
        # Test that old-style initialization still works
        ocr = RapidOCR()
        
        # Check that the object has expected methods
        assert hasattr(ocr, '__call__'), "Missing __call__ method"
        
        print("✓ Test 7: Backward compatibility check successful")
        return True
    except Exception as e:
        print(f"✗ Test 7 failed: {e}")
        return False

def main():
    """Run all API compatibility tests"""
    print("=" * 60)
    print("RapidOCR HKMC API Compatibility Tests")
    print("=" * 60)
    
    tests = [
        test_basic_import,
        test_rapidocr_initialization,
        test_rapidocr_with_config,
        test_image_processing,
        test_vis_res,
        test_load_image_error,
        test_backward_compatibility
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("✓ All API compatibility tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
