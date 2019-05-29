import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch-es",
    version="0.0.1",
    author="Vladimir Shulyak",
    author_email="vladimir@shulyak.net",
    description="Double Seasonal Exponential Smoothing using PyTorch + ES-RNN capabilities on top",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vshulyak/torch-es",
    packages=setuptools.find_packages(),
    install_requires=[
        'torch>=1.0.0',
        'numpy>=1.16.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
