import setuptools

with open("README.md") as file:
    long_description = file.read()

setuptools.setup(
    name="event_warping",
    version="0.0.1",
    author="ICNS, Alexandre Marcireau",
    author_email="alexandre.marcireau@gmail.com",
    description="Post-process ISS data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "event_stream >= 1.4",
        "h5py >= 3.0",
        "matplotlib >= 3.0",
        "numpy >= 1.20",
        "pillow >= 9.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=["event_warping"],
)
