#include "bert.h"
#include "ggml.h"

#include <unistd.h>
#include <stdio.h>
#include <vector>

int main(int argc, char ** argv) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    bert_params params;
    if (bert_params_parse(argc, argv, params) == false) {
        return 1;
    }

    int64_t t_load_us = 0;

    bert_ctx * bctx;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if ((bctx = bert_load_from_file(params.model, params.use_cpu)) == nullptr) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model);
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    int64_t t_start_us = ggml_time_us();

    // tokenize the prompt
    int N = bert_n_max_tokens(bctx);
    bert_tokens tokens = bert_tokenize(bctx, params.prompt, N);

    int64_t t_mid_us = ggml_time_us();
    int64_t t_token_us = t_mid_us - t_start_us;

    for (auto & tok : tokens) {
        printf("%d -> %s\n", tok, bert_vocab_id_to_token(bctx, tok));
    }
    printf("\n");

    // run the embedding
    const int n_embd = bert_n_embd(bctx);
    std::vector<float> embeddings(n_embd);
    bert_forward(bctx, tokens, embeddings.data(), params.n_threads);

    int64_t t_end_us = ggml_time_us();
    int64_t t_eval_us = t_end_us - t_mid_us;
    
    printf("[ ");
    for(auto e : embeddings) {
        printf("%1.4f, ", e);
    }
    printf("]\n");

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n");
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:    token time = %8.2f ms / %.2f ms per token\n", __func__, t_token_us/1000.0f, t_token_us/1000.0f/tokens.size());
        printf("%s:     eval time = %8.2f ms / %.2f ms per token\n", __func__, t_eval_us/1000.0f, t_eval_us/1000.0f/tokens.size());
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    return 0;
}
