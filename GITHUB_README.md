# Hunyuan3D Glasses Generation

Generate 3D glasses models from 2D images using Hunyuan3D 2.0.

![Glasses Generation Demo](https://github.com/amin8452/Hunyuan3D-Glasses-Generation/raw/main/demo/demo.gif)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/amin8452/Hunyuan3D-Glasses-Generation.git
cd Hunyuan3D-Glasses-Generation

# Install dependencies
pip install -r requirements.txt
```

### Generate 3D Glasses

```bash
python run_glasses_generation.py --input_image your_image.jpg --output_model glasses.glb
```

That's it! Your 3D glasses model will be saved as `glasses.glb`.

## Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (for GPU acceleration)
- 8GB+ RAM

## Options

```
--input_image    Path to the input image (required)
--output_model   Path for the output 3D model (default: output.glb)
--output_dir     Directory for output files (default: outputs)
--visualize      Visualize the generated 3D model
```

## Examples

Generate 3D glasses from an image:

```bash
python run_glasses_generation.py --input_image examples/glasses1.jpg --output_model my_glasses.glb
```

Generate and visualize:

```bash
python run_glasses_generation.py --input_image examples/glasses2.jpg --visualize
```

## Viewing 3D Models

You can view the generated GLB files using:
- Online viewers like [glTF Viewer](https://gltf-viewer.donmccurdy.com/)
- 3D software like Blender
- AR/VR applications that support glTF/GLB formats

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project adapts the Hunyuan3D 2.0 model developed by Tencent for generating 3D glasses models from 2D images.
