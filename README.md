# Density Invariant Contrast Maximization for Neuromorphic Earth Observations [CVPRW2023]

Code for [Density Invariant Contrast Maximization for Neuromorphic Earth Observations](https://arxiv.org/abs/2304.14125).

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

> **Note:**  The publication of raw data is currently undergoing a review process by the Air Force Research Lab (AFRL). For further information or inquiries regarding the publication status and availability of the data, please feel free to contact us directly.


## Summary of the algorithm:

The proposed method takes events as input, which can be very dense and noisy, and the temporal window can be arbitrarily large. Without any prior knowledge of the camera motion, it applies an analytical geometric-based correction on the warped image to ensure high contrast when the events are optimally aligned. By employing this approach, the method guarantees a convex loss surface, particularly in dense scenes.

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

In **`scripts/DensityInvariantCMax.py`**:

Specify the parameters and run the code.

```
DensityInvariantCMax(filename=EVENTS[-1], 
                     heuristic="weighted_variance", 
                     velocity_range=(-30, 30), 
                     resolution=30, 
                     ratio=0.0000001, 
                     tstart=0, 
                     tfinish = 50e6,
                     read_path="./data/",
                     save_path="./img/")
```

This outputs the loss landscape across vx and vy. This is the difference between the landscapes if you used the `heuristic="variance"` (Left) and `heuristic="weighted_variance"` (Right)

<p align="center">
  <img alt="Light" src="./img/landscape_before_after.png" width="70%">
</p> 

Alternatinely you can choose not to compute the variance for every single $v_x$ and $v_y$ and use an optimisation algorithm to search for the best speed value by changing the `SOLVER` and `HEURISTIC` options in `scripts/find_theta.py`.

To learn more about the analytical solution of the variance, please see the jupyter notebook [analysis_1d.ipynb](scripts/analysis_1d.ipynb) and [analysis_2d.ipynb](scripts/analysis_2d.ipynb)

To see how the analytical solution was applied on real-world ISS data: [DensityInvariantCMax.py](scripts/DensityInvariantCMax.py)
