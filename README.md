
# Detecting Dengue in Flight - README

Welcome to the **Detecting Mosquito in Flight** project. Follow the steps below to download, install, and run the application.

### Download and Install

1. **Download the executable**:
   - Download the `.exe` file from the installation folder [here](https://drive.google.com/drive/folders/1fI0PbEfFs53tcndoB7yPG9Pp4paDLnJN).
   
2. **Download the Trained Model**:
   - Download the trained model for the user interface, named `Mosquito Detection Model.pt`, from this GitHub repository.
   
3. **Optional**: If you wish to custom-train your model, follow the instructions at the [Ultralytics GitHub Repository](https://github.com/ultralytics/ultralytics) for guidance on custom model training.

---

### Installation and Setup

#### Step 1: Install Conda (if you don’t have it)
- Install **Anaconda** from the [official site](https://www.anaconda.com/products/individual).

#### Step 2: Create a Conda Environment
- Open **Anaconda Prompt** and create a new environment by typing:

    ```bash
    conda create --name mosquito_detection python=3.9
    ```

- Activate the environment:

    ```bash
    conda activate mosquito_detection
    ```

#### Step 3: Install Required Libraries
- Once the environment is active, install the prerequisite libraries from the `requirements.txt` file located in the repository. Run the following command:

    ```bash
    pip install -r requirements.txt
    ```

---

### Running the Application

1. Navigate to the `Code` folder using the terminal or Anaconda prompt.
   
2. Run the classification code after adding 3-dimensional data paths:

    ```bash
    python main.py
    ```

