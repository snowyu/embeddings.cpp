# bert.cpp

[ggml](https://github.com/ggerganov/ggml) inference of BERT embedding models including [SentenceTransformers (sbert.net)](https://sbert.net/), [BGE series](https://huggingface.co/BAAI/bge-base-en-v1.5), and others. High quality sentence embeddings in pure C++ and Python (bindings).

This repo is a fork of original [bert.cpp](https://github.com/skeskinen/bert.cpp) as well as [embeddings.cpp](https://github.com/xyzhang626/embeddings.cpp). Thanks to both of you!

### Install

Fetch this respository and download the submodules with
```sh
git submodule update --init --recursive
```

To fetch models from `huggingface`  and convert them to `gguf` format run the following
```sh
cd models
python download-repo.py BAAI/bge-base-en-v1.5 # or any other model
python convert-to-ggml.py BAAI/bge-base-en-v1.5 f16
python convert-to-ggml.py BAAI/bge-base-en-v1.5 f32
```

### Build

To build the dynamic library for usage from Python
```sh
cmake -B build .
make -C build
```

If you're compiling for GPU, you may need to run something like
```sh
cmake -DCMAKE_CUDA_COMPILER=/home/doug/programs/cuda/bin/nvcc -DGGML_CUBLAS=ON -B build .
```

On some distros, you also need to specifiy the host C++ compiler. To do this, I suggest setting the `CUDAHOSTCXX` environment variable to your C++ bindir.

### Excecute

All executables are placed in `build/bin`. To run inference on a given text, run
```sh
build/bin/main -m models/bge-base-en-v1.5/ggml-model-f16.gguf -p "Hello world"
```
To force CPU usage, add the flag `-c`.

### Python

You can also run everything through Python, which is particularly useful for batch inference. For instance,
```python
import bert
mod = bert.BertModel('models/bge-base-en-v1.5/ggml-model-f16.gguf')
emb = mod.embed(batch)
```
where `batch` is a list of strings and `emb` is a `numpy` array of embedding vectors.

### Quantize

You can quantize models with the command
```sh
build/bin/quantize models/bge-base-en-v1.5/ggml-model-f32.gguf models/bge-base-en-v1.5/ggml-model-q8_0.gguf q8_0
```
or whatever your desired quantization level is. Currently supported values are: `q8_0`, `q5_0`, `q5_1`, `q4_0`, and `q4_1`. You can then pass these model files directly to `main` as above.
