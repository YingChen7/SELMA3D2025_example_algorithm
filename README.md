# SELMA3D 2025: Example for building a container image of your algorithm for Preliminary Development Phase Task 2 w/o SSL 

### Step 1: Implement your solution  
* In [requirements.txt](requirements.txt), list the packages required for your solution.
* In [inference.py](inference.py), `load_image_file_as_array` function will automatically load the testing image once you submit your algorithm container. Do not change the [image reading](inference.py#L53) and [saving](inference.py#L108) parts.
  
* Modify the [processing part](inference.py#L61) in inference.py file to preprocess the input image and define the model.
* Put the resources required for prediction to the [resource folder](resources) such as model checkpoints, then modify the [resource part](inference.py#L73) in inference.py file to load the resources.
* Modify the [prediction part](inference.py#L92) in inference.py file, replacing it with your solution to make a prediction for the loaded image array.

### Step 2: Test locally
* Call the [do test_run](do_test_run.sh) bash script using the command:
        ```./do_test_run.sh```

  This will start the inference and reads from [/test/input](/test/input) and outputs to /test/output.
### Step 3: Save the container image for Grand Challenge submission
*  Call the [do save](do_save.sh) bash script using the command:
        ```./do_save.sh```

   This will create a container image of the algorithm.
### Step 4 : Use the github repo for Grand Challenge submission (optional) 
*  Fork this repository to a new repository under your GitHub account.
*  In your repository, complete Step 1 and Step 2 to implement and test your solution.
*  Follow the instructions below for submitting to the Grand Challenge (you may need to copy and paste the link into your browser to access it):
   https://grand-challenge.org/documentation/linking-a-github-repository-to-your-algorithm/