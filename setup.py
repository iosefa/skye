import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="skye",
    version="0.0.5",
    author="Iosefa Percival",
    author_email="ipercival@gmail.com",
    description="360-degree image analysis for forest ecology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iosefa/skye",
    project_urls={
        "Bug Tracker": "https://github.com/iosefa/skye/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'ipython>=8.12.0',
        'matplotlib>=3.7.1',
        'numpy>=1.24.2',
        'opencv-python>=4.7.0.72',
        'pandas>=2.0.0',
        'Pillow>=9.5.0',
        'scikit-image>=0.20.0',
        'scikit-learn>=1.2.2',
        'scipy>=1.10.1',
        'tqdm>=4.65.0'
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8"
)