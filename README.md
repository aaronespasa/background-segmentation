# Background Segmentation

## Set-Up Environment 🌲

### Install the necessary dependencies

1. **Create a virtual environment**
   ```sh
   python -m venv venv
   ```

2. **Activate the environment**
   - **UNIX-based OSs:**
     ```sh
     source venv/bin/activate
     ```
   - **Windows:**
     ```cmd
     .\venv\Scripts\activate
     ```

3. **Install PyTorch**
   - Visit the [PyTorch - Get Started](https://pytorch.org/get-started/locally/) page to get the correct installation commands for your setup.
   - Example command (may vary based on your hardware and Python version):
     ```sh
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```

4. **Install additional requirements**
   ```sh
   pip install -r requirements.txt
   ```

### Download, filter, and process the dataset

1. **Create a Kaggle API Token**
   - Follow the instructions in [this video](https://www.youtube.com/watch?v=L-CzBRXefXY) to create a "kaggle.json" file.

2. **Move the API Token to the appropriate location**
   - **UNIX-based OSs:**
     ```sh
     mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/
     ```
   - **Windows:**
     ```cmd
     move kaggle.json C:\Users\<Windows-username>\.kaggle\
     ```

3. **Install the Kaggle CLI**
   ```sh
   pip install -q kaggle
   ```

4. **Download and extract the dataset**
   - **Download command:**
     ```bash
     kaggle datasets download -d aaronespasa/matting-human-small-dataset
     ```
   - **Extract the dataset:**
     - **UNIX-based OSs:**
       ```sh
       ./data.sh
       ```
     - **Windows:**
       ```cmd
       ./data.sh  # Execute in Git Bash
       ```

## Train the Model 🛠

- **Run the Jupyter Notebook:**
  ```sh
  jupyter notebook training.ipynb
  ```
