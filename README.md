## Dataset and Usage Notes

The dataset used in this project belongs to Prof. Dr. Alpan Bek. Due to data sharing restrictions original raw dataset is not shared with this repository.

As a result, `preprocessing.ipynb` notebook which was originally used to process the raw data can not be executed without access to raw files. However, it is included for illustrate the preprocessing steps applied before training.

The preprocessed dataset used in this study is available via an external link (https://drive.google.com/drive/folders/1f_oKUhIuQQqCYChH171NfNvVo5N5Kruq?usp=sharing). After downloaded, make sure to update `data_path` variable inside the `main.ipynb` notebook accordingly to point to your local dataset folder.

This project is implemented based on [1] and portions of the original code from [1] are adapted. Link: https://github.com/


## Project File Descriptions

- `preprocessing.ipynb` = Jupyter notebook used for preprocessing the raw Raman spectroscopy data (not directly usable without raw files)
- `main.ipynb` = main notebook to run training, evaluation and visualization of results
- `resnet.py`= defines custom 1D ResNet architecture used for the regression task
- `data.py` =loading and optional augmentation of the dataset for training.
- `augment.py`= augmentation functions and tools for both online and offline data augmentation.
- `training.py` = training loop, loss tracking, and performance evaluation