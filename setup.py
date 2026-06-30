"""
Setup script for the wtpf (Wind Turbine Power Forecasting) package.

Install in editable/development mode with:
    pip install -e .
"""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""

setup(
    name="wtpf",
    version="1.0.0",
    description="CNN-Attention-BiLSTM Hybrid Deep Learning Model for Wind Turbine Power Forecasting using SCADA Data",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Md. Uzzal Mia, Sajib Debnath, Arindam Kishor Biswas, Md. Sarwar Hosain, Tetsuya Shimamura",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
