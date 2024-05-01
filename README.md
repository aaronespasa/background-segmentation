# Teeth Segmentation

## Set-Up Environment 🌲 
### Install the necessary dependencies
1. Create an environmental file:
```sh
$ python -m venv venv
```

2.1. Activate the environment using Windows:
```sh
$ venv\Scripts\activate
```

2.2. Activate the environment using Linux or MacOS:
```sh
$ source venv/bin/activate
```

3. Install PyTorch locally getting the commands from ([PyTorch - Get Started](https://pytorch.org/get-started/locally/)):
```sh
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. Install the requirements:
```sh
$ pip install -r requirements.txt
```

### Download the dataset, filter it and process it

1. Download the dataset from [this link](https://drive.google.com/file/d/1JDc8Y6nIiyzUw7vB-d5Qqv9MqvNtJAXG/view?usp=sharing).
2. Create a folder named `data` in the root directory of the project.
3. Put the folders "annotations" and "images" inside the `data` folder.

## Train the model 🛠
The model can be trained by running the Jupyter Notebook `training.ipynb`.