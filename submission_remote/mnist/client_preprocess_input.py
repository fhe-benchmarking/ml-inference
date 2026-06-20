import torch
import sys
from lattica_query.serialization import api_serialization_utils
import submission_utils
local_file_paths = submission_utils.init(int(sys.argv[1]))

# Read inputs from file
examples = []
with open(local_file_paths.TEST_DATA_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        pix = [float(p) for p in line.split(" ")]
        if len(pix) != 28 * 28:
            raise ValueError(f"Expected 784 pixels, got {len(pix)}")
        examples.append(pix)

pt = torch.tensor(examples).reshape(-1, 28*28)
# Apply MNIST normalization: (pixel - mean) / std
pt = (pt - 0.1307) / 0.3081
pt_ser = api_serialization_utils.dumps_proto_tensor(pt)

with open(local_file_paths.PT_PATH, "wb") as f:
    f.write(pt_ser)
