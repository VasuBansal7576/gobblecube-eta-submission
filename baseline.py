#!/usr/bin/env python
"""Compatibility wrapper.

The starter trained an XGBoost baseline from this file. The submission model is
trained by train.py; keeping this wrapper makes the original run order still
work for reviewers.
"""

from __future__ import annotations

from train import main


if __name__ == "__main__":
    main()
