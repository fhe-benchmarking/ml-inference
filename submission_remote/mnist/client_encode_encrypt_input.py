import pickle
import sys

import lattica_query.query_toolkit as toolkit_interface
from lattica_query.serialization.hom_op_pb2 import ClientModel as ProtoClientModel

import submission_utils
local_file_paths = submission_utils.init(int(sys.argv[1]))

# Read data from local filesystem required for encoding and encrypting
with open(local_file_paths.PT_PATH, "rb") as f:
    pt_ser = f.read()

context_ser = pickle.load(open(local_file_paths.PATH_CONTEXT, "rb"))
model_ser = pickle.load(open(local_file_paths.PATH_HOM_SEQ, "rb"))
sk =      pickle.load(open(local_file_paths.PATH_SK, "rb"))

model_proto = ProtoClientModel()
model_proto.ParseFromString(model_ser)

preprocess_block_ser = model_proto.preprocess_block.SerializeToString()
pt_axis_external = model_proto.pt_axis_external if model_proto.HasField("pt_axis_external") else None

pt_enc = toolkit_interface.apply_client_block(preprocess_block_ser, context_ser, pt_ser)
ct_batch = toolkit_interface.enc(context_ser, sk, pt_enc, pack_for_transmission=True, n_axis_external=pt_axis_external)
pickle.dump(ct_batch, open(local_file_paths.get_ct_upload_path("batch"), "wb"))
