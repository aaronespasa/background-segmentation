# Background Segmentation

## Set-Up Environment 🌲 
### Install the necessary dependencies
1. Create an environmental file:
```sh
$ python -m venv venv
```

2. Activate the environment:
   <ol type="a">
     <li><b>UNIX-based OSs:</b> <code>source venv/bin/activate</code>.</li>
     <li><b>Windows:</b> <code>.\venv\Scripts\activate</code>.</li>
   </ol>

3. Install PyTorch locally getting the commands from ([PyTorch - Get Started](https://pytorch.org/get-started/locally/)):
```sh
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. Install the requirements:
```sh
$ pip install -r requirements.txt
```

### Download the dataset, filter it, and process it

1. Create a Kaggle API Token following the instructions of [this video](https://www.youtube.com/watch?v=L-CzBRXefXY). This will create a "kaggle.json" file.
2. Move this file to its appropriate location:
   <ol type="a">
     <li><b>UNIX-based OSs:</b> <code>~/.kaggle/kaggle.json</code>.</li>
     <li><b>Windows:</b> <code>C:\Users\&lt;Windows-username&gt;\.kaggle\kaggle.json</code>.</li>
   </ol>
3. Then install the kaggle CLI:
```bash
pip install -q kaggle
```
4. Download the dataset:
```bash
kaggle datasets download -d aaronespasa/matting-human-small-dataset
```
5. Extract the dataset:
   <ol type="a">
     <li><b>UNIX-based OSs:</b> <code>./data.sh</code>.</li>
     <li><b>Windows:</b> Open a Git Bash and then execute the following <code>./data.sh</code>.</li>
   </ol>  

## Train the model 🛠
The model can be trained by running the Jupyter Notebook `training.ipynb`.
