import sys
import pickle

import submission_utils
local_file_paths = submission_utils.init(int(sys.argv[1]))

client = submission_utils.get_lattica_client(local_file_paths)

ek = pickle.load(open(local_file_paths.PATH_EK, "rb"))
temp_filename = 'ek.lpk'
with open(temp_filename, 'wb') as handle:
    handle.write(ek)

client.upload_evaluation_key_file(temp_filename)
