from setuptools import setup, find_packages

setup(
    name="ep5244",  # Nouveau nom du package
    version="0.2",
    description="Material for the 52445EP Labs",  # Description
    author="Pascal Bianchi",                      # Auteur
    url="https://github.com/pascalbianchi/54244EP",  # URL du dépôt
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "torch","torchvision"
    ],
)

