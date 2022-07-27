# event_warping

Event warping experiments.

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
