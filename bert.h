#ifndef BERT_H
#define BERT_H

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <map>

#define BERT_API __attribute__ ((visibility ("default")))

#ifdef __cplusplus
extern "C" {
#endif

//
// data types
//

typedef int32_t bert_token;
typedef std::vector<bert_token> bert_tokens;
typedef std::vector<bert_tokens> bert_batch;
typedef std::string bert_string;
typedef std::vector<bert_string> bert_strings;

//
// data structures
//

// default hparams (all-MiniLM-L6-v2)
struct bert_hparams {
    int32_t n_vocab = 30522;
    int32_t n_max_tokens = 512;
    int32_t n_embd = 256;
    int32_t n_intermediate = 1536;
    int32_t n_head = 12;
    int32_t n_layer = 6;
    float_t layer_norm_eps = 1e-12;
};

struct bert_layer {
    // normalization
    struct ggml_tensor *ln_att_w;
    struct ggml_tensor *ln_att_b;

    struct ggml_tensor *ln_out_w;
    struct ggml_tensor *ln_out_b;

    // attention
    struct ggml_tensor *q_w;
    struct ggml_tensor *q_b;
    struct ggml_tensor *k_w;
    struct ggml_tensor *k_b;
    struct ggml_tensor *v_w;
    struct ggml_tensor *v_b;

    struct ggml_tensor *o_w;
    struct ggml_tensor *o_b;

    // ff
    struct ggml_tensor *ff_i_w;
    struct ggml_tensor *ff_i_b;

    struct ggml_tensor *ff_o_w;
    struct ggml_tensor *ff_o_b;
};

struct bert_vocab {
    std::vector<std::string> tokens;

    std::map<std::string, bert_token> token_to_id;
    std::map<std::string, bert_token> subword_token_to_id;

    std::map<bert_token, std::string> _id_to_token;
    std::map<bert_token, std::string> _id_to_subword_token;
};

struct bert_model {
    bert_hparams hparams;

    // embeddings weights
    struct ggml_tensor *word_embeddings;
    struct ggml_tensor *token_type_embeddings;
    struct ggml_tensor *position_embeddings;
    struct ggml_tensor *ln_e_w;
    struct ggml_tensor *ln_e_b;

    std::vector<bert_layer> layers;
};

struct bert_ctx {
    bert_model model;
    bert_vocab vocab;

    // ggml context
    struct ggml_context * ctx_data;

    // compute metadata
    std::vector<uint8_t> buf_compute_meta;

    // memory buffers to evaluate the model
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t weights_buffer = NULL;
    ggml_backend_buffer_t compute_buffer = NULL;
    ggml_allocr * compute_alloc = NULL;
};

//
// main api
//

BERT_API struct bert_ctx * bert_load_from_file(
    const char * fname,
    bool use_cpu
);

BERT_API void bert_allocate_buffers(
    bert_ctx * ctx,
    int32_t n_max_tokens,
    int32_t batch_size
);

BERT_API void bert_deallocate_buffers(bert_ctx * ctx);
BERT_API void bert_free(bert_ctx * ctx);

BERT_API ggml_cgraph * bert_build_graph(
    bert_ctx * ctx,
    bert_batch batch
);

BERT_API void bert_forward_batch(
    bert_ctx * ctx,
    bert_batch tokens,
    float * embeddings,
    int32_t n_thread
);

BERT_API void bert_encode_batch(
    struct bert_ctx * ctx,
    bert_strings texts,
    float * embeddings,
    int32_t n_threads
);

BERT_API void bert_encode_batch_c(
    struct bert_ctx * ctx,
    const char ** texts,
    float * embeddings,
    int32_t n_input,
    int32_t n_threads
);

BERT_API bert_tokens bert_tokenize(
    struct bert_ctx * ctx,
    bert_string text,
    int32_t n_max_tokens
);

BERT_API int32_t bert_tokenize_c(
    struct bert_ctx * ctx,
    const char * text,
    int32_t * output,
    int32_t n_max_tokens
);

BERT_API void bert_forward(
    struct bert_ctx * ctx,
    bert_tokens tokens,
    float * embeddings,
    int32_t n_thread
);

BERT_API void bert_encode(
    struct bert_ctx * ctx,
    bert_string text,
    float * embeddings,
    int32_t n_threads
);

BERT_API int32_t bert_n_embd(bert_ctx * ctx);
BERT_API int32_t bert_n_max_tokens(bert_ctx * ctx);

BERT_API const char* bert_vocab_id_to_token(bert_ctx * ctx, bert_token id);

#ifdef __cplusplus
}
#endif

#endif // BERT_H
