# Next Steps for Hunyuan3D Glasses Generation

This document outlines the implementation of the next steps for adapting Hunyuan3D 2.0 for 3D glasses generation.

## 1. Data Collection

We've implemented a comprehensive data collection pipeline in `scripts/data_collection.py` that:

- Creates a structured dataset with train/val/test splits
- Supports downloading from various eyewear retailers (placeholder implementation)
- Categorizes glasses by style, material, and color
- Applies basic filtering to ensure quality
- Generates metadata for each image

### Usage

```bash
python scripts/data_collection.py --output_dir data --input_dir your_glasses_images
```

### Key Features

- **Automatic categorization**: Uses computer vision to categorize glasses styles
- **Balanced dataset creation**: Ensures representation across different styles
- **Quality filtering**: Removes low-quality or non-glasses images
- **Metadata generation**: Creates detailed metadata for each image

### Data Structure

```
data/
├── train/
│   └── images/
├── val/
│   └── images/
├── test/
│   └── images/
├── raw/
│   ├── round/
│   ├── square/
│   └── ...
└── metadata.json
```

## 2. Fine-tuning

We've implemented a flexible fine-tuning pipeline in `scripts/fine_tuning.py` that:

- Loads a pre-trained Hunyuan3D 2.0 model
- Supports different fine-tuning strategies (full, last layers, progressive)
- Implements various optimizers and learning rate schedulers
- Tracks metrics and saves checkpoints

### Usage

```bash
python scripts/fine_tuning.py \
    --train_data data/train \
    --val_data data/val \
    --pretrained_model models/hunyuan3d_2.0.pt \
    --fine_tuning_strategy last_layers \
    --batch_size 8 \
    --epochs 50 \
    --lr 1e-5
```

### Fine-tuning Strategies

1. **Full**: Fine-tune the entire model
2. **Last Layers**: Freeze the base model, fine-tune only the adaptation layers
3. **Progressive**: Start with adaptation layers, then gradually unfreeze more layers

### Monitoring

- Training progress is logged to TensorBoard
- Checkpoints are saved regularly
- The best model is saved based on FID score

## 3. Hyperparameter Optimization

We've implemented a hyperparameter optimization pipeline in `scripts/hyperparameter_optimization.py` that:

- Uses Optuna to find the best hyperparameters
- Optimizes learning rate, batch size, optimizer, scheduler, etc.
- Supports early stopping to save computation
- Generates detailed reports of the optimization process

### Usage

```bash
python scripts/hyperparameter_optimization.py \
    --train_data data/train \
    --val_data data/val \
    --pretrained_model models/hunyuan3d_2.0.pt \
    --n_trials 20 \
    --epochs 10 \
    --train_best
```

### Optimized Hyperparameters

- Learning rate
- Batch size
- Weight decay
- Optimizer (AdamW, SGD, RMSprop)
- Scheduler (cosine, onecycle)
- Fine-tuning strategy
- Momentum (for SGD and RMSprop)

### Output

- Detailed JSON report of all trials
- Visualization of hyperparameter importance
- Option to automatically train a model with the best hyperparameters

## 4. Comprehensive Evaluation

We've implemented a comprehensive evaluation pipeline in `scripts/comprehensive_evaluation.py` that:

- Evaluates the model on various glasses styles
- Compares generated models with ground truth 3D models
- Generates detailed reports and visualizations
- Performs user-oriented metrics (wearability, realism)

### Usage

```bash
python scripts/comprehensive_evaluation.py \
    --model_path models/fine_tuned/best_model.pt \
    --test_data data/test \
    --output_dir outputs/evaluation
```

### Evaluation Metrics

1. **Style Accuracy**: How well the model preserves the style of the input glasses
2. **3D Metrics**:
   - Chamfer distance
   - Edge loss
   - Laplacian smoothing
   - Normal consistency
3. **Texture Quality**:
   - Texture similarity
   - Color accuracy
   - Texture detail
4. **Wearability**:
   - Width-to-height ratio
   - Temple length ratio
   - Overall ergonomics

### Output

- Comprehensive JSON report
- Visualizations comparing original images, generated models, and ground truth
- Confusion matrix for style accuracy
- Bar charts for 3D metrics and wearability

## Integration with Existing Code

These new scripts integrate seamlessly with the existing codebase:

- They use the same model architecture (`models/hunyuan3d_adapted.py`)
- They leverage the same dataset utilities (`utils/dataset.py`)
- They follow the same evaluation approach as `scripts/evaluate.py`
- They can be used alongside the existing preprocessing and generation scripts

## Next Steps

1. **Implement real data collection**: Replace placeholder implementations with actual web scraping or dataset acquisition
2. **Refine 3D metrics**: Implement more sophisticated 3D comparison metrics
3. **User study**: Conduct a user study to evaluate the perceived quality of generated glasses
4. **Deployment**: Create a web interface for easy generation of 3D glasses from images

## Requirements

Additional dependencies for these scripts are included in the updated `requirements.txt`:

```
optuna>=2.10.0
scikit-learn>=1.0.0
seaborn>=0.11.0
pandas>=1.3.0
pytorch3d>=0.7.0
```
