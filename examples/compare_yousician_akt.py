"""Compatibility entry point for Yousician comparisons."""
try:
    from .compare_akt_fmkc import main
except ImportError:
    from compare_akt_fmkc import main

if __name__ == "__main__":
    main()
