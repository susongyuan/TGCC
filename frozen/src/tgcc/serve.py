"""Active server: success-distribution plus retrieval-conditioned ActionExpert.

Legacy rule/evaluation artifacts cannot be loaded by this entry point.
"""
from .distribution_serve import main

if __name__ == "__main__":
    main()
