#!/bin/bash

echo "📁 Creating folders..."
mkdir -p data images models checkpoints

echo "📊 Running EDA..."
python3 src/eda.py

echo "🤖 Training model..."
python3 src/train_model.py

echo "🧪 Evaluating model..."
python3 src/evaluate_model.py

echo "✅ All steps complete!"
