import sys
import json
import torch
import numpy as np
from pathlib import Path
from gguf import GGUFWriter, GGMLQuantizationType

from transformers import AutoModel, AutoTokenizer

# primay usage
if len(sys.argv) < 2:
    print('Usage: convert-to-ggml.py dir-model [use-f32]\n')
    print('  ftype == 1 -> float32')
    print('  ftype == 0 -> float16')
    sys.exit(1)

# output in the same directory as the model
dir_model = Path(sys.argv[1])
ftype = int(sys.argv[2]) if len(sys.argv) > 2 else 0

# map from ftype to string
ftype_str = 'f32' if ftype == 1 else 'f16'
fname_out = dir_model / f'ggml-model-{ftype_str}.bin'

# heck if the directory existsc
if not dir_model.exists():
    print(f'Directory {dir_model} does not exist.')

# load hf modle data
with open(dir_model / 'tokenizer.json', 'r', encoding='utf-8') as f:
    encoder = json.load(f)

with open(dir_model / 'config.json', 'r', encoding='utf-8') as f:
    hparams = json.load(f)

with open(dir_model / 'vocab.txt', 'r', encoding='utf-8') as f:
    vocab = f.readlines()

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(dir_model)
model = AutoModel.from_pretrained(dir_model, low_cpu_mem_usage=True)

print()
print('HPARAMS:')
print(hparams)
print()

# start to write GGUF file
gguf_writer = GGUFWriter(fname_out, 'bert')

# write metadata
file_type = GGMLQuantizationType.F32 if ftype == 1 else GGMLQuantizationType.F16
gguf_writer.add_name('BERT')
gguf_writer.add_description('GGML BERT model')
gguf_writer.add_file_type(file_type)

# write kv flags
gguf_writer.add_uint32('vocab_size', hparams['vocab_size'])
gguf_writer.add_uint32('max_position_embedding', hparams['max_position_embeddings'])
gguf_writer.add_uint32('hidden_size', hparams['hidden_size'])
gguf_writer.add_uint32('intermediate_size', hparams['intermediate_size'])
gguf_writer.add_uint32('num_attention_heads', hparams['num_attention_heads'])
gguf_writer.add_uint32('num_hidden_layers', hparams['num_hidden_layers'])
gguf_writer.add_float32('layer_norm_eps', hparams['layer_norm_eps'])

# write vocab
gguf_writer.add_token_list(vocab)

# write tensors
for name, data in model.state_dict().items():
    # skip some params
    if name in ['embeddings.position_ids', 'pooler.dense.weight', 'pooler.dense.bias']:
        continue
    else:
        print(f'{name}: {data.dtype} {list(data.shape)}')

    # convert weights to f16?
    dtype = torch.float32 if ftype == 1 else torch.float16
    data = data.to(dtype).numpy()

    # header
    gguf_writer.add_tensor(name, data)

# execute and close writer
gguf_writer.write_header_to_file()
gguf_writer.write_kv_data_to_file()
gguf_writer.write_tensors_to_file()
gguf_writer.close()
