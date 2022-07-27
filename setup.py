import builtins
import setuptools
import setuptools.extension
import setuptools.command.build_ext
import sys

with open("README.md") as file:
    long_description = file.read()


class build_ext(setuptools.command.build_ext.build_ext):
    def finalize_options(self):
        setuptools.command.build_ext.build_ext.finalize_options(self)
        builtins.__NUMPY_SETUP__ = False  # type: ignore
        import numpy

        self.include_dirs.append(numpy.get_include())


extra_args = []
if sys.platform == "linux":
    extra_args += ["-std=c++11"]
elif sys.platform == "darwin":
    extra_args += ["-std=c++11", "-stdlib=libc++"]

setuptools.setup(
    name="event_warping",
    version="0.0.1",
    author="ICNS, Alexandre Marcireau",
    author_email="alexandre.marcireau@gmail.com",
    description="Post-process ISS data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    setup_requires=['numpy'],
    install_requires=[
        "cmaes >= 0.8.2",
        "event_stream >= 1.4.1",
        "h5py >= 3.7.0",
        "matplotlib >= 3.5.2",
        "numpy >= 1.23.1",
        "pillow >= 9.2.0",
        "scipy >= 1.8.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=["event_warping"],
    ext_modules=[
        setuptools.extension.Extension(
            "event_warping_extension",
            language="c++",
            sources=["event_warping_extension/event_warping_extension.cpp"],
            include_dirs=[],
            libraries=[],
            extra_compile_args=extra_args,
            extra_link_args=extra_args,
        ),
    ],
    cmdclass={"build_ext": build_ext},
)