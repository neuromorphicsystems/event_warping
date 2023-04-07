# Density Invariant Contrast Maximisation for Neuromorphic Earth Observations

This contains the implementation of the paper "Density Invariant Contrast Maximisation for Neuromorphic Earth Observations".

**Summary of the algorithm:** 

Take input data in which the variance landscape has multiple extrema where one of those correspond to the correct motion parameters (Left), it then applies volume based correction on the warped image (Middle) to produce a landscape with a single peak which correspond to the motion parameters (Right). The entire network make use of the standard motion compensation algorithm but our method makes it resilient to data that is extremely dense and noisy.

<p align="center">
  <img alt="Light" src="./img/before_correction.gif" width="23%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="./img/framework.png" width="40%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="./img/after_correction.gif" width="24%">
</p>


## ISS Datasets
Below is a short snippet of each data. From Right to Left: El Salvador, Houston, Brittany, Mexico, Washington, Spain, Summatra, UK, Egypt and Panama. 

<p align="center">
  <img alt="Light" src="./img/20220121a_Salvador_2022-01-21_20~58~34_NADIR.h5_cut_exponential_frametime=20ms_tau=200ms.gif" width="20%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="./img/20220217_Houston_IAH_1_2022-02-17_20-28-02_NADIR_cut_exponential_frametime=20ms_tau=200ms.gif" width="20%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="./img/20220125_Brittany_211945_2022-01-25_21-21-18_NADIR_cut_exponential_frametime=20ms_tau=200ms.gif" width="20%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Light" src="./img/20220125_Mexican_Coast_205735_2022-01-25_20~58~52_NADIR.h5_cut_exponential_frametime=20ms_tau=200ms.gif" width="20%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="./img/20220201_DIA_201410_2022-02-01_20-15-58_NADIR_cut_exponential_frametime=20ms_tau=200ms.gif" width="20%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="./img/20220127_Biscay_Spain_Med_211912_2_2022-01-27_21-53-58_NADIR_cut_exponential_frametime=20ms_tau=200ms.gif" width="20%">
 &nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Light" src="./img/20220217_Sumatra_Night_2_2022-02-17_21_cut_exponential_frametime=20ms_tau=200ms.gif" width="20%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="./img/20230119_4_UK_Nadir_Night_2023-01-19_20~25~10_NADIR_cut_exponential_frametime=20ms_tau=200ms.gif" width="20%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="./img/FN034HRTEgyptB_NADIR_cut4_exponential_frametime=20ms_tau=200ms.gif" width="20%"> 
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="./img/20220124_201028_Panama_2022-01-24_20_12_11_NADIR.h5_cut_exponential_frametime=20ms_tau=200ms.gif" width="20%"> 
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
    Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 2072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    choco install -y visualstudio2019buildtools
    choco install -y visualstudio2019-workload-vctools
    ```

    See https://chocolatey.org for details.

## Usage

Python files in _scripts_ implement two different methods to estimate the visual speed (px/s) for the ISS recordings:

1. The implementation of the standard Contrast Maximisation framework which uses the variance as an objective function.
2. The method proposed in this paper which divides the Contrast Maximisation process into multiple peicewise fuctions and apply multiplicative weight to remove the global maxima to preserving the correct motion parameter.

To run the code it is recommended to have the have the recording in either `.es` or `.h5`. If your events data are in `.mat` format, then use `scripts/mattoes.py` to convert them to`.es`. Otherwise, please refer to [command_line_tool](https://github.com/neuromorphic-paris/command_line_tools) and [event_stream](event_stream) to learn more about how to convert and process your event data in `.es`.


Before running the example, create a directory called _recordings_ in the _event_warping_ directory and place `20220124_202028_Panama_2022-01-24_20~12~11_NADIR.h5` in _recordings_.

```
python3 example.py
```
Expected output:

<p align="center">
  <img alt="Light" src="./img/Panama_2022-01-24_20_12_11_NADIR_21.49_-0.74.png" width="90%">
</p>

To produce the entire loss landscape similarly to Table 1, run the following:

**`scripts/space.py`**

Contains the code for running contrast maximization with two different heuristics approach: (1) Standard CMax and (2) the method proposed in the paper.

`HEURISTIC = "variance"` will run the standard contrast maximisation algorithm and save the loss landscape in .json and .png in your chosen directory.

`HEURISTIC = "weighted_variance"` will the run the method proposed in the paper which include a volumetric-based correction on the loss landscape. Both .json and .png will be saved in your chosen directory.

`HEURISTIC = "max"` is a simple custom objective function that takes the maximum as the loss. Not used in the paper and it is not investigated further. This can be a guide if you wish to include a new custom objective function along with the other methods.

Below is an example of using the **variance** (left) and the **weighted_variance** (right) methods. The weighted variance method will automatically remove the global maximum and keep a single local maximum that belongs to the correct motion parameters. 

<p align="center">
  <img alt="Light" src="./img/landscape_before_after.png" width="70%">
</p>

Alternatinely you can choose not to compute the variance for every single $v_x$ and $v_y$ and directly optimise for the best speed value by changing the `SOLVER` and `HEURISTIC` options in `find_theta.py`.

**`scripts/find_theta.py`**


Use this script to use an optimiser to quickly find the correct motion parameters, without doing an exhaustive search across the entire motion space. The script includes a gradient-based (BFGS) and non-gradiend-based (Nedler-mead) optimiser as well as other optimisation methods for testing. If you wish to add another optimisation method or/and heuristic, see `event_warping` class.

# An example of the motion corrected output from the ISS (Over Suez Canal, Egypt)

<p align="center">
  <img alt="Light" src="./img/suezcanalfinalresults.gif" width="70%">
</p>

