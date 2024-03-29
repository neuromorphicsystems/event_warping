# Density Invariant Contrast Maximization for Neuromorphic Earth Observations [CVPRW2023]

Code for **Density Invariant Contrast Maximization for Neuromorphic Earth Observations**.

```bibtex
@InProceedings{arjaDensityInvariantCMax2023,
    author    = {Arja, Sami and Marcireau, Alexandre and Balthazor, Richard L. and McHarg, Matthew G. and Afshar, Saeed and Cohen, Gregory},
    title     = {Density Invariant Contrast Maximization for Neuromorphic Earth Observations},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {3983-3993}
}
```
<table align="center">
  <tr>
    <td align="center" style="border:none;">
      <a href="https://arxiv.org/pdf/2304.14125.pdf" target="_blank">
        <img src="./img/arxiv.jpeg" alt="arXiv" width="50" style="padding:10px; background-color: #f5f5f5; border-radius: 10px; box-shadow: 2px 2px 12px #aaa;">
      </a>
    </td>
    <td align="center" style="border:none;">
      <a href="./img/2023CVPRW_DICMaxNEO_poster.pdf" target="_blank">
        <img src="./img/poster_img.png" alt="Poster" width="68" style="padding:10px; background-color: #f5f5f5; border-radius: 10px; box-shadow: 2px 2px 12px #aaa;">
      </a>
    </td>
  </tr>
</table>




> **Note:**  The publication of the raw data is currently undergoing a review process. Specific data can be released through written requests in the interim. Any requests should be directed to Associate Professor Gregory Cohen at Western Sydney University.

A simple noisy and dense data is provided in `data/simple_noisy_events_with_motion.es` to run the algorithm.

## Summary

This paper addresses the problem of the noise intolerance in Contrast Maximization framework and proposed an analytical solution to it. The solution was evaluated on dense event data from the ISS.

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

This documentation provides instructions for utilizing the code and exploring various functionalities. Follow the steps below to get started.

#### Dataset format
The input events are assumed to be either in the [event_stream (.es)](https://github.com/neuromorphicsystems/event_stream) or .txt format. Please refer to the [loris](https://github.com/neuromorphic-paris/loris) library to convert to/from .es format.
The .txt file should be structured like: `txyp`

You don't have to specify the file format (e.g. .es or .txt), just specify the file name and a function will figure the format and use the suitable data loader.

#### Testing

run [earth_mapping.py](earth_mapping.py) to generate motion compensated image using the density invariant contrast maximization method. Recommended for quick testing.

#### Generating Loss Landscape
To generate the loss landscape w.r.t the motion parameters $v_x$ and $v_y$, use `density_invariant_cmax.py`:

First adjust the parameters as necessary.:

```
DensityInvariantCMax(filename="path_to_your_event_data",
                     heuristic=OBJECTIVE[1],
                     velocity_range=(-50, 50),
                     resolution=50,
                     ratio=0.0,
                     tstart=0,
                     tfinish=10e6, #for 10 second
                     read_path="data/",
                     save_path="img/")
```
and run `python density_invariant_cmax.py`

This outputs the loss landscape across vx and vy. This is the difference between the landscapes if you used the `heuristic="variance"` (Left: **two solutions**) and `heuristic="weighted_variance"` (Right: **single solution**)

<p align="center">
  <img alt="Light" src="./img/landscape_before_after.png" width="70%">
</p> 


#### Solvers

Alternatively you can choose not to compute the variance for every single $v_x$ and $v_y$ and use an optimisation algorithm to search for the best speed value by changing the `solver` and `heuristic` options in `scripts/optimise_cmax.py`.

```
OptimizeCMax(filename=EVENTS[0],
             heuristic=OBJECTIVE[0],
             solver=solver[0],
             initial_speed=200, #a random value between (-200,200) for vx and vy
             tfinish=10e6, #for 10 second
             ratio=0.0,
             read_path="data/",
             save_path="img/")
```

and run `python scripts/optimise_cmax.py`

#### Analytical Modeling

For a detailed understanding of the analytical modeling approach, refer to the provided Jupyter notebooks:
- [analysis_1d.ipynb](scripts/analysis_1d.ipynb)
- [analysis_2d.ipynb](scripts/analysis_2d.ipynb)

These notebooks contain explanations and code snippets showcasing the analytical approach.

#### For the actual implementation on the real-world ISS data: 
run [implementation.py](scripts/implementation.py)

#### Extending Heuristics and Solvers

To implement a new heuristic and/or a new solver, you can edit the file inside: `event_warping/*.py`, or if you want to implement them in C++ then you can edit the file inside `event_warping_extension/*.cpp`. However, if you edit the .cpp file, you have to run this command in the terminal to enable the changes:

```python3 -m pip install -e .```

#### Graphical Interface
To visualize the change in the geometries in the image of the warped events w.r.t various motion candidates $v_x$ and $v_y$, a GUI is provided. First install vispy and pyqt:

```pip install vispy```
```pip install PyQt5```

Then from the terminal run:
```python3 shear.py```