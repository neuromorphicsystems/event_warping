# Density Invariant Contrast Maximisation for Neuromorphic Earth Observations [CVPRW2023]

This contains the implementation of the paper [Density Invariant Contrast Maximisation for Neuromorphic Earth Observations](https://arxiv.org/abs/2304.14125).

> **Note:** ISS dataset will be available soon.


## Summary of the algorithm:

Our proposed method takes events that have multiple peaks in its variance landscape, with one of them representing the correct motion parameters (left). It then applies an analytical geometric-based correction on the warped image (middle), to produces a landscape with a single peak that corresponds to the correct motion parameters (right). The method takes no prior of the camera motion and it does not require the event rate.

<p align="center">
  <img alt="Light" src="./img/before_correction.gif" width="26%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="./img/proposed_method.png" width="35%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="./img/after_correction.gif" width="26%">
</p>

## Algorithm in action

<p align="center">
  <img alt="Light" src="./img/density_invariant_CMX.gif" width="80%">
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

Citation:
```bibtex
@article{arja2023density,
  title={Density Invariant Contrast Maximization for Neuromorphic Earth Observations},
  author={Arja, Sami and Marcireau, Alexandre and Balthazor, Richard L and McHarg, Matthew G and Afshar, Saeed and Cohen, Gregory},
  journal={arXiv preprint arXiv:2304.14125},
  year={2023}
}
```