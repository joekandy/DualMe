from setuptools import setup, find_packages

setup(
    name="dualme",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "gradio>=3.50.0",
        "opencv-python>=4.5.0",
        "scikit-image>=0.19.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.5.0",
        "requests>=2.31.0",
        "gdown>=4.7.1"
    ],
    python_requires=">=3.8",
) 