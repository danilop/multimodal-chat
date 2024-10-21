#!/bin/sh
echo "Deleting images..."
rm -rf ./Images
echo "Deleting output files..."
rm -rf ./Output
echo "Resetting index..."
./multimodal_chat.py --reset-index