#ifndef BERT_H
#define BERT_H

#include "ggml.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string>
#include <vector>

#define BERT_API __attribute__ ((visibility ("default")))

#ifdef __cplusplus
extern "C" {
#endif

struct bert_ctx;

typedef int32_t bert_token;
typedef std::vector<bert_token> bert_tokens;
typedef std::vector<bert_tokens> bert_batch;
typedef std::string bert_string;
typedef std::vector<bert_string> bert_strings;

// Main api, does both tokenizing and evaluation

BERT_API struct bert_ctx * bert_load_from_file(
    const char * fname,
    int32_t batch_size,
    bool use_cpu
);

BERT_API void bert_free(
    bert_ctx * ctx
);

BERT_API ggml_cgraph * bert_build_graph(
    bert_ctx * ctx,
    bert_batch batch);

BERT_API void bert_forward_batch(
    bert_ctx * ctx,
    bert_batch tokens,
    float * embeddings,
    int32_t n_threads);

// n_batch_size - how many to process at a time
// n_inputs     - total size of texts and embeddings arrays
BERT_API void bert_encode_batch(
    struct bert_ctx * ctx,
    bert_strings texts,
    float * embeddings,
    int32_t n_threads);

// takes c-style list of strings
BERT_API void bert_encode_batch_c(
    struct bert_ctx * ctx,
    const char ** texts,
    float * embeddings,
    int32_t n_input,
    int32_t n_threads);

// Api for separate tokenization & eval

BERT_API bert_tokens bert_tokenize(
    struct bert_ctx * ctx,
    bert_string text,
    int32_t n_max_tokens);

BERT_API void bert_forward(
    struct bert_ctx * ctx,
    bert_tokens tokens,
    float * embeddings,
    int32_t n_threads);

BERT_API void bert_encode(
    struct bert_ctx * ctx,
    bert_string text,
    float * embeddings,
    int32_t n_threads);

BERT_API int32_t bert_n_embd(bert_ctx * ctx);
BERT_API int32_t bert_n_max_tokens(bert_ctx * ctx);

BERT_API const char* bert_vocab_id_to_token(bert_ctx * ctx, bert_token id);

#ifdef __cplusplus
}
#endif

#endif // BERT_H
