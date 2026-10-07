"""
setup.py
========
Checks the environment and prints a sanity-check line, as required by the
course setup step ("post OK + your numpy/scipy/matplotlib versions").

Usage:
    python setup.py
"""

import sys


def main():
    try:
        import numpy
        import scipy
        import matplotlib
    except ImportError as e:
        print(f"FAIL: missing dependency ({e})")
        print("Install with:  pip install numpy scipy matplotlib")
        sys.exit(1)

    print("OK")
    print(f"  python      {sys.version.split()[0]}")
    print(f"  numpy       {numpy.__version__}")
    print(f"  scipy       {scipy.__version__}")
    print(f"  matplotlib  {matplotlib.__version__}")


if __name__ == "__main__":
    main()
