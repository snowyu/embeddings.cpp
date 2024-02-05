# bert.cpp

This is a [ggml](https://github.com/ggerganov/ggml) implementation of the BERT embedding architecture. It supports inference on CPU, CUDA and Metal in floating point and a wide variety of quantization schemes. Includes Python bindings for batched inference.

This repo is a fork of original [bert.cpp](https://github.com/skeskinen/bert.cpp) as well as [embeddings.cpp](https://github.com/xyzhang626/embeddings.cpp). Thanks to both of you!

### Install

Fetch this repository then download submodules and install packages with
```sh
git submodule update --init --recursive
pip install -r requirements.txt
```

To fetch models from `huggingface`  and convert them to `gguf` format run the following
```sh
cd models
python download.py BAAI/bge-base-en-v1.5 # or any other model
python convert.py bge-base-en-v1.5 f16
python convert.py bge-base-en-v1.5 f32
```

### Build

To build the dynamic library for usage from Python
```sh
cmake -B build .
make -C build -j
```

If you're compiling for GPU, you should run
```sh
cmake -DGGML_CUBLAS=ON -B build .
make -C build -j
```
On some distros, you also need to specify the host C++ compiler. To do this, I suggest setting the `CUDAHOSTCXX` environment variable to your C++ bindir.

And for Apple Metal, you should run
```sh
cmake -DGGML_METAL=ON -B build .
make -C build -j
```

### Execute

All executables are placed in `build/bin`. To run inference on a given text, run
```sh
# CPU / CUDA
build/bin/main -m models/bge-base-en-v1.5/ggml-model-f16.gguf -p "Hello world"

# Metal
GGML_METAL_PATH_RESOURCES=build/bin/ build/bin/main -m models/bge-base-en-v1.5/ggml-model-f16.gguf -p "Hello world"
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
