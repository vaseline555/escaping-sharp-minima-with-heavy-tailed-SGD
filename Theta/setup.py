from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="theta",
    version="0.1.0",
    packages=find_packages(),
    description="Truncated heavy-tailed noise injection for SGD optimizers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    install_requires=["torch>=2.0.0"],
)
