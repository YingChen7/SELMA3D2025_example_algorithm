"""
The following is a simple example algorithm.

It is meant to run within a container.

To run the container locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

from pathlib import Path
import json
import torch
from glob import glob
import SimpleITK
from functools import partial
import numpy
from monai.networks.nets import UNETR
from monai.inferers import sliding_window_inference

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def run():
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()

    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        ("light-sheet-3d-microscopy",): interf0_handler,
    }[interface_key]

    # Call the handler
    return handler()


def interf0_handler():
    # Read the input
    input_light_sheet_3d_microscopy = load_image_file_as_array(
        location=INPUT_PATH / "images/light-sheet-3d-microscopy",
    )

    # Process the inputs: any way you'd like, here we show-case torch
    _show_torch_cuda_info()

    # Example how to set torch to use the GPU (if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    input = torch.from_numpy(input_light_sheet_3d_microscopy.astype(numpy.float32)).unsqueeze(0).unsqueeze(0).to(device)
    model = UNETR(in_channels=1, out_channels=1, 
                  img_size=(128, 128,128), feature_size=32, 
                  hidden_size=768, mlp_dim=3072, 
                  num_heads=12, pos_embed='perceptron',
                  norm_name='instance', conv_block=True, 
                  res_block=True, dropout_rate=0.0)
    # Some additional resources might be required, include these in one of two ways.

    # Option 1: part of the Docker-container image: resources/
    resource_dir = Path("/opt/app/resources")
    if device.type =='cpu':
        model_dict = torch.load(resource_dir.joinpath('model_final.pt'),weights_only=False, map_location='cpu')["state_dict"]
    else:
        model_dict = torch.load(resource_dir.joinpath('model_final.pt'),weights_only=False)["state_dict"]
    
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    # Option 2: upload them as a separate tarball to Grand Challenge (go to your Algorithm > Models). The resources in the tarball will be extracted to `model_dir` at runtime.
    '''
    model_dir = Path("/opt/ml/model")
    with open(
        model_dir / "a_tarball_subdirectory" / "some_tarball_resource.txt", "r"
    ) as f:
        print(f.read())
    '''
    # For now, let us make bogus predictions

    model_inferer = partial(sliding_window_inference, roi_size = [128, 128, 128],
                            sw_batch_size = 1,
                            predictor = model,
                            overlap = 0.5,
                            mode = 'gaussian')
    with torch.no_grad():
        output_logits = model_inferer(input)
    prob = torch.sigmoid(output_logits)
    
    output_contiguous_biological_structure  = prob[0, 0].detach().cpu().numpy()
    output_contiguous_biological_structure  = (output_contiguous_biological_structure >0.5).astype(numpy.int8)
    print(output_contiguous_biological_structure .shape)
    

    # Save your output
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/contiguous-biological-structure",
        array=output_contiguous_biological_structure,
    )

    return 0


def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def write_array_as_image_file(*, location, array):
    location.mkdir(parents=True, exist_ok=True)

    # You may need to change the suffix to .tif to match the expected output
    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
