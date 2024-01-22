# bert.cpp

[ggml](https://github.com/ggerganov/ggml) inference of BERT neural net architecture with pooling and normalization from embedding models including [SentenceTransformers (sbert.net)](https://sbert.net/), [BGE series](https://huggingface.co/BAAI/bge-base-en-v1.5) and others.
High quality sentence embeddings in pure C++ (with C API).

This repo is a fork of original [bert.cpp](https://github.com/skeskinen/bert.cpp).

In this fork, we have added support for:

+ Multilingual tokenizer inlcuding Asian languages and latin languages.
+ Support real batch inference.
+ Support current SOTA embedding model [BGE series](https://huggingface.co/BAAI/bge-base-en-v1.5).


## Description
The main goal of `bert.cpp` is to run the BERT model using 4-bit integer quantization on CPU

* Plain C/C++ implementation without dependencies
* Inherit support for various architectures from ggml (x86 with AVX2, ARM, etc.)
* Choose your model size from 32/16/4 bits per model weigth
* all-MiniLM-L6-v2/BGE with 4bit quantization is only 14MB. Inference RAM usage depends on the length of the input
* Sample cpp server over tcp socket and a python test client
* Benchmarks to validate correctness and speed of inference

## Limitations & TODO

+ Update to the latest ggml lib and gguf format.
+ Current memory management is not good enough, should be improved using lastest ggml api.

## Usage

### Checkout the ggml submodule
```sh
git submodule update --init --recursive
```

### Get models
```sh
pip install -r requirements.txt
cd models
python download-repo.py BAAI/bge-base-en-v1.5 # or any other model
sh run_conversions.sh bge-base-en-v1.5
```

### Test tokenizer

In this fork we support multilingual tokenizer, you can test different model's tokenzier by:

```sh
bash test_tokenizer.sh bge-base-en-v1.5
```

This script will tokenize the content in `models/test_prompts.txt` with both huggingface tokenizer and this tokenizer, and compare the results. You can add more content in the `models/test_prompts.txt` to test more cases. Note that orignal `bge-small-zh-v1.5` tokenizer (not this repo) is some problematic, refer to this [issue](https://github.com/xyzhang626/embeddings.cpp/issues/1) for more details.

### Build
To build the dynamic library for usage from e.g. Python:
```sh
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
make
cd ..
```

### Converting models to ggml format
Converting models is similar to llama.cpp. Use models/convert-to-ggml.py to make hf models into either f32 or f16 ggml models. Then use ./build/bin/quantize to turn those into Q4_0, 4bit per weight models.

There is also models/run_conversions.sh which creates all 4 versions (f32, f16, Q4_0, Q4_1) at once.
```sh
cd models
# Clone a model from hf
python download-repo.py USERNAME/MODEL_NAME
# Run conversions to 4 ggml formats (f32, f16, Q4_0, Q4_1)
sh run_conversions.sh MODEL_NAME
```
