import pickle
import sys
from lattica_query.serialization import api_serialization_utils
import lattica_query.query_toolkit as toolkit_interface

import submission_utils
local_file_paths = submission_utils.init(int(sys.argv[1]))

context = pickle.load(open(local_file_paths.PATH_CONTEXT, "rb"))
sk = pickle.load(open(local_file_paths.PATH_SK, "rb"))

ct_res = pickle.load(open(local_file_paths.get_ct_download_path("batch"), "rb"))
res = toolkit_interface.dec(context, sk, ct_res, False)
res = api_serialization_utils.load_proto_tensor(res)
if res.ndim == 1:
    res = res.unsqueeze(0)
results = res.argmax(axis=1).tolist()

with open(local_file_paths.PREDICTIONS_PATH, "w") as f:
    f.write("\n".join(str(int(r)) for r in results) + "\n")
