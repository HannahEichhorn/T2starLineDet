# Physics-Aware Motion Simulation for T2*-Weighted Brain MRI

Paper accepted at [MICCAI 2023 SASHIMI workshop](https://2023.sashimi-workshop.org/).


All computations were performed using Python 3.8.12 and PyTorch 1.13.0.


## Contents:

- `motion_simulation`: simulating realistic motion artefacts in T2*w GRE MRI data
- `line_det_network`: training and testing a k-space line detection network, using the [IML-CompAI Framework](https://github.com/compai-lab/iml-dl) 
- `evaluation`: evaluating the proposed method, using the [medutils package](https://github.com/khammernik/medutils) for the TV reconstruction


## Prerequisites:

1. Create a virtual environment with the required packages:
    ```
    cd ${TARGET_DIR}/T2starLineDet
    conda env create -f conda_environment.yaml
    source activate t2star_linedet *or* conda activate t2star_linedet
    ```

2. Install pytorch with cuda:
    ```
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    pip install torchinfo
    conda install -c conda-forge pytorch-lightning
    ```

3. For setting up wandb please refer to the [IML-CompAI Framework](https://github.com/compai-lab/iml-dl).


## Steps to reproduce the analysis:

1)  Motion Simulation:

    i) Run `motion_simulation/PCA_motion_curves_CV.py` to perform a prinicpal component analysis on the motion data

    ii)  Run `motion_simulation/Scan_order.py` to extract the acquisition order of all k-space lines from a raw file (ISMRMRD format)
    
    iii) Run `motion_simulation/Prepare_config.py` to generate configuration files needed for running the motion simulation

    iv)  Run `motion_simulation/Simulate_Motion_Whole_Dataset.py` to run the motion simulations

2) Line Detection Network:

    Follow the instructions in `line_det_network/README.md`

3) Final Evaluations:

    i) Run `evaluation/EvaluatePredictions.py` to evaluate the network performance and reconstruct example motion corrected images



## Illustration of the motion simulation procedure:
![Simulation_overview](/visualisation_architecture.pdf?raw=true "Overview of motion simulation")


## Illustration of the network architecture:
![Architecture_overview](/visualisation_motion_simulation.pdf?raw=true "Architecture of k-space line detection network")


## Results
Classification performance is decreasing for decreasing simulation thresholds:

![Results_performance](/results_performance_diff_thr.pdf?raw=true "Test accuracy, rates of non-detected (ND) and wrongly-detected (WD) lines for varying thresholds in the motion simulation of train and test data.")

Weighted reconstructions show subtly redued artefacts:

![Results_example_recons](/results_example_recons.pdf?raw=true "Demonstration of weighted reconstructions with TV regularisation for simulated data with very mild and slightly stronger motion (top/bottom row, mean displacement during whole scan: 0.50/0.89 mm).")

For a more detailed description of the results please refer to our paper.


## Citation
If you use this code, please cite

```
@InProceedings{eichhorn2023deep,
      title={Physics-Aware Motion Simulation for {T2*}-Weighted Brain {MRI}}, 
      author={Hannah Eichhorn and Kerstin Hammernik and Veronika Spieker and Samira M. Epp and Daniel Rueckert and Christine Preibisch and Julia A. Schnabel},
      booktitle="Simulation and Synthesis in Medical Imaging. SASHIMI 2023. Lecture Notes in Computer Science",
      year={2023},
      publisher={Springer International Publishing}
}
```