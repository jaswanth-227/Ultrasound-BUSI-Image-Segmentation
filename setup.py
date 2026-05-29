from setuptools import setup, find_packages

setup(
    name="HEMF-Net",
    version="1.0.0",
    author="Manoj Kumar Sunkara",
    description="Hierarchical Enhanced Multi-scale Fusion Network for Breast Ultrasound Image Segmentation",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "opencv-python",
        "matplotlib",
        "scikit-learn",
        "scikit-image",
        "torch",
        "torchvision",
        "torchaudio",
        "timm",
        "transformers",
        "einops",
        "albumentations",
        "tqdm",
        "Pillow",
    ],
    python_requires=">=3.10",
)