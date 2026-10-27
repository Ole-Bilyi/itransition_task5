#!/usr/bin/env bash
set -o errexit

# Install Chromium for Kaleido/Plotly image export
apt-get update
apt-get install -y chromium

# Install Python dependencies
pip install -r requirements.txt