from setuptools import setup, find_packages

setup(
    name="profession-classifier",
    version="0.1.0",
    description="CNN-based classifier for identifying professionals from images",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "pillow>=8.2.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0"
    ],
    python_requires=">=3.7",
)

