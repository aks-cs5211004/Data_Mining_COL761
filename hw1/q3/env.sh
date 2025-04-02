#!/bin/bash


echo "Installing required packages..."
pip install networkx rustworkx numpy joblib scikit-learn

if [ -f "./Binaries/gaston" ]; then
    chmod +x ./Binaries/gaston
    echo "gaston binary found and set to executable."
else
    echo "Warning: gaston binary not found in ./Binaries/"
fi
echo "Library installation complete."
