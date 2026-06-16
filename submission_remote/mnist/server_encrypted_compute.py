import json
import pickle
import sys
from lattica_query import performance_utils

import submission_utils
local_file_paths = submission_utils.init(int(sys.argv[1]))

client = submission_utils.get_lattica_client(local_file_paths)

ct = pickle.load(open(local_file_paths.get_ct_upload_path("batch"), "rb"))
ct_res = client.worker_api.apply_hom_pipeline(ct)
pickle.dump(ct_res, open(local_file_paths.get_ct_download_path("batch"), "wb"))

# Parse and save server timing report
server_timing = client.worker_api.get_last_timing()
server_report_for_harness = performance_utils.server_timing_report(server_timing)
with open(local_file_paths.SERVER_TIMES_PATH, "w") as f:
    json.dump(server_report_for_harness, f)
