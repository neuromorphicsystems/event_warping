import builtins
import pathlib
import platform
import setuptools
import setuptools.extension
import setuptools.command.build_ext
import setuptools.command.develop

with open("README.md") as file:
    long_description = file.read()


class Develop(setuptools.command.develop.develop):
    user_options = setuptools.command.develop.develop.user_options + [
        ("with-extension", None, "compile the C++ extension"),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.with_extension = False

    def run(self):
        if self.with_extension:
            compiler = platform.python_compiler()
            extra_compile_args = []
            extra_link_args = []
            if compiler.startswith("Clang"):
                extra_compile_args += ["-std=c++11", "-stdlib=libc++"]
                extra_link_args += ["-std=c++11", "-stdlib=libc++"]
            elif compiler.startswith("GCC"):
                extra_compile_args += ["-std=c++11"]
                extra_link_args += ["-std=c++11"]
            builtins.__NUMPY_SETUP__ = False  # type: ignore
            import numpy

            self.distribution.ext_modules = [  # type: ignore
                setuptools.extension.Extension(
                    "event_warping_extension",
                    language="c++",
                    sources=["event_warping_extension/event_warping_extension.cpp"],
                    include_dirs=[numpy.get_include()],
                    libraries=[],
                    extra_compile_args=extra_compile_args,
                    extra_link_args=extra_link_args,
                ),
            ]
        else:
            pathlib.Path("event_warping_extension.cpython-39-darwin.so").unlink(
                missing_ok=True
            )
            pass
        super().run()


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
    cmdclass={
        "develop": Develop,  # type: ignore
    },
)
