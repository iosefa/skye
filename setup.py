import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="skye",
    version="0.0.2",
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
    packages=setuptools.find_packages(),
    python_requires=">=3.8"
)