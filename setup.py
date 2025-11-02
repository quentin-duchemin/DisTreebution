from setuptools import setup, find_packages
import pathlib


here = pathlib.Path(__file__).parent
try:
    long_description = (here / "package_docs" / "README.md").read_text(encoding="utf-8")
except Exception:
    long_description = "DisTreebution — regression-tree uncertainty quantification and conformal prediction utilities."

setup(
    name="distreebution",
    version="0.1.0",
    description="Regression-tree based conformal prediction and uncertainty quantification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Quentin Duchemin",
    author_email="quentin.duchemin@epfl.ch",
    url="",
    packages=find_packages(exclude=("package_docs", "docs", "examples", "tests")),
    python_requires='>=3.10',
    install_requires=[
        "numpy",
        "pandas",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "matplotlib",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
