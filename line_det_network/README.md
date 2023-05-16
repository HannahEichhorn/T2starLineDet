# Deep Learning-Based Detection of Motion-Affected k-Space Lines for T2*-Weighted MRI


## Line Detection network

For this part of the work, code from the following framework was used: [IML-CompAI Framework](https://github.com/compai-lab/iml-dl) (commit 199101d).

In this folder, a copy of the above code is added with the following changes:

- `iml-dl/projects/kspace_line_detection/`: contains additional project-specific code
- `iml-dl/weights/kspace_line_detection/`: contains the weights of the trained model (for different simulation thresholds)
- `iml-dl/core/Trainer.py/`: lines 86-88 are commented out and performed with correct input size in `iml-dl/projects/kspace_line_detection/Trainer.py`


## Training:

1) Use the file `iml-dl/projects/kspace_line_detection/CreateDataLinks.py` to create symbolic links under `iml-dl/data/links_to_data/`.

2) Adapt `iml-dl/projects/kspace_line_detection/config_train_3D_thr_class.yaml` to correct settings (i.e. input and output folders)

3) Run the following commands:
```
# go to right directory and activate the conda environment
cd path_to_code/T2starLineDet/line_det_network/iml-dl/
conda activate t2star_linedet

# launch the experiment
echo 'Starting Script'
python -u ./core/Main.py --config_path ./projects/kspace_line_detection/config_train_3D_thr_class.yaml
```
