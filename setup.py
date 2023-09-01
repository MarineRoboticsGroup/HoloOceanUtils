import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="HoloOceanUtils",
    version="0.0.1",
    author="Wells Crosby",
    author_email="wcrosby@mit.edu",
    description="Holoocean Utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License"
    ),
    install_requires=[
        'numpy',
    ],
)