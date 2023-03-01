# Dense Piecewise Contrast Maximisation for Space-based Observations

This contains the implementation of the paper "Dense Piecewise Contrast Maximisation for Space-based Observations".

<!-- <p align="center">
      <img src="./img/correction_algorithm.svg" align="left">
      <img src="./img/correction_algorithm.svg">
      <img src="./img/correction_algorithm.svg" align="right"> -->
<!-- </p> -->

<p align="center">
  <img alt="Light" src="./img/before_correction.gif" width="23%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="./img/framework.png" width="44%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="./img/after_correction.gif" width="24%">
</p>

## Install

```sh
git clone git@github.com:neuromorphicsystems/event_warping.git # alternatively, download as zip
cd event_warping
python3 -m pip install -e .
```

The installation process compiles the event_warping_extension, which is required to speed up event projection calculations. If the installation fails, you may need to install a C++ compiler with one of the following methods:

-   **Ubuntu**

    ```sh
    sudo apt install -y build-essentials
    ```

-   **macOS**

    ```sh
    xcode-select --install
    ```

-   **Windows**

    Start > Windows Powershell > Right click > Run as Administator

    ```powershell
    Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    choco install -y visualstudio2019buildtools
    choco install -y visualstudio2019-workload-vctools
    ```

    See https://chocolatey.org for details.

## Usage

To run the code it is recommended to have the have the recording in either `.es` or `.h5`. If your events data are in `.mat` format, then use `scripts/mattoes.py` to convert them to`.es`. Otherwise, please refer to [command_line_tool](https://github.com/neuromorphic-paris/command_line_tools) and [event_stream](event_stream) to learn more about how to convert and process your event data in `.es`.


Before running the example, create a directory called _recordings_ in the _event_warping_ directory and place `20220124_201028_Panama_2022-01-24_20~12~11_NADIR.h5` in _recordings_.

```
python3 example.py
```
Expected output:
![image info](./img/Panama_2022-01-24_20_12_11_NADIR_21.49_-0.74.png)

## Scripts

Python files in _scripts_ implement two different methods to estimate the visual speed (px/s) for the ISS recordings:

1. The implementation of the standard Contrast Maximisation framework which uses the variance as an objective function.
2. The method proposed in this paper which divides the Contrast Maximisation process into multiple peicewise fuctions and apply multiplicative weight to remove the global maxima to preserving the correct motion parameter.

_scripts/configuration.py_ defines the parameters (file name, intial velocity...) used by the other script files.

-   _optimize_speed_contrast_maximization.py_ calculates the velocity that maximizes image contrast. Run `python3 optimize_speed_contrast_maximization.py --help` to list available optimization methods.
-   _plot_space_1d.py_ generates a 1D optimization space plot. _space_1d.py_ must be run first.
-   _plot_space.py_ generates a 2D optimization space plot. _space.py_ must be run first.
-   _project.py_ generates a contrast maximized image and its histogram using the velocity in _configuration.py_.
-   _space_1d.py_ calculates the image contrast for every vx (resp. vy) at constant vy (resp. vx).
- _peicewise_objective_algo_space.py_ contains the original implementation of the peicewise contrast maximisation algorithm. It takes input `.es` or `.h5` file and output the corrected loss landscape. There is not optimisation included in this, instead it does an exhaustive search for the parameter but in the corrected space. This takes a long time to run.
-   _space.py_ calculates the image contast for every pair (vx, vy). You can switch between the standard method and proposed approach using the following steps:

```sh
    weight
```

There is not optimisation included in this, instead it does an exhaustive search for the parameter but in the corrected space. This takes a long time to run.

## References

## Install

```sh
git clone git@github.com:neuromorphicsystems/event_warping.git # alternatively, download as zip
cd event_warping
python3 -m pip install -e .
```

The installation process compiles the event_warping_extension, which is required to speed up event projection calculations. If the installation fails, you may need to install a C++ compiler with one of the following methods:

-   **Ubuntu**

    ```sh
    sudo apt install -y build-essentials
    ```

-   **macOS**

    ```sh
    xcode-select --install
    ```

-   **Windows**

    Start > Windows Powershell > Right click > Run as Administator

    ```powershell
    Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    choco install -y visualstudio2019buildtools
    choco install -y visualstudio2019-workload-vctools
    ```

    See https://chocolatey.org for details.

## Example

Before running the example, create a directory called _recordings_ in the _event_warping_ directory and place _20220124_201028_Panama_2022-01-24_20~12~11_NADIR.h5_ in _recordings_.

```
python3 example.py
```

## Scripts

Python files in _scripts_ implement different methods to estimate the visual speed (px/s) of ISS recordings and generate figures to understand why the optimization process (contrast maximization) fails in some cases.

_scripts/configuration.py_ defines the parameters (file name, intial velocity...) used by the other script files.

-   _optimize_speed_contrast_maximization.py_ calculates the velocity that maximizes image contrast. Run `python3 optimize_speed_contrast_maximization.py --help` to list available optimization methods.
-   _plot_space_1d.py_ generates a 1D optimization space plot. _space_1d.py_ must be run first.
-   _plot_space.py_ generates a 2D optimization space plot. _space.py_ must be run first.
-   _project.py_ generates a contrast maximized image and its histogram using the velocity in _configuration.py_.
-   _space_1d.py_ calculates the image contrast for every vx (resp. vy) at constant vy (resp. vx).
-   _space.py_ calculates the image contast for every pair (vx, vy). This script takes a long time to run.

## Papers

A Unifying Contrast Maximization Framework for Event Cameras, with Applications to Motion, Depth, and Optical Flow Estimation

https://ieeexplore.ieee.org/document/8578505

https://arxiv.org/pdf/1804.01306.pdf
