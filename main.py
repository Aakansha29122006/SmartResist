#!/usr/bin/env python
# coding: utf-8
"""
SmartResist — Main Entry Point
================================
Usage:
    python main.py --train          # Train the model
    python main.py --serve          # Start Flask server
    python main.py --train --serve  # Train then serve
    python main.py                  # Default: train then serve
"""

import argparse
import sys
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)


def run_training():
    """Execute the full training pipeline."""
    from train_model import main as train_main
    train_main()


def run_server():
    """Start the Flask prediction server."""
    from app import app, load_model_artifacts

    print("============================================================")
    print("|   SmartResist -- Flask Prediction Server                 |")
    print("============================================================\n")

    load_model_artifacts()
    app.run(debug=False, port=5000, host='0.0.0.0')


def main():
    parser = argparse.ArgumentParser(
        description="SmartResist — AI Antibiotic Resistance Prediction System"
    )
    parser.add_argument(
        '--train', action='store_true',
        help='Run the full ML training pipeline'
    )
    parser.add_argument(
        '--serve', action='store_true',
        help='Start the Flask prediction server'
    )

    args = parser.parse_args()

    # Default behavior: train then serve
    if not args.train and not args.serve:
        args.train = True
        args.serve = True

    if args.train:
        run_training()

    if args.serve:
        run_server()


if __name__ == '__main__':
    main()
