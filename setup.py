"""Setup configuration for H_MODEL_Z framework."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = (this_directory / "config" / "requirements.txt").read_text().splitlines()
requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

setup(
    name="hmodelz",
    version="1.0.0",
    author="H_MODEL_Z Team",
    author_email="contact@hmodelz.dev",
    description="Enterprise-grade performance optimization framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zkaedii/qsim",
    project_urls={
        "Bug Tracker": "https://github.com/zkaedii/qsim/issues",
        "Documentation": "https://github.com/zkaedii/qsim/docs",
        "Source Code": "https://github.com/zkaedii/qsim",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "pre-commit>=2.15.0",
        ],
        "gpu": [
            "torch>=1.9.0",
            "cupy>=9.0.0",
        ],
        "distributed": [
            "mpi4py>=3.1.0",
            "dask>=2021.6.0",
        ],
        "viz": [
            "streamlit>=1.10.0",
            "plotly>=5.0.0",
            "altair>=4.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hmodelz=hmodelz.hmodelz_cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "optimization",
        "performance",
        "quantum-simulation",
        "blockchain",
        "ai",
        "enterprise",
        "hamiltonian",
        "defi",
    ],
)
