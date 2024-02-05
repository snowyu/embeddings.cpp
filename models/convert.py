import sys
import json
import torch

from pathlib import Path
from gguf import GGUFWriter, GGMLQuantizationType, TokenType
from transformers import AutoModel, AutoTokenizer
from sentencepiece import SentencePieceProcessor

KEY_PAD_ID = 'tokenizer.ggml.padding_token_id'
KEY_UNK_ID = 'tokenizer.ggml.unknown_token_id'
KEY_BOS_ID = 'tokenizer.ggml.bos_token_id'
KEY_EOS_ID = 'tokenizer.ggml.eos_token_id'
KEY_SUBWORD_PREFIX = 'tokenizer.ggml.subword_prefix'

class SimpleVocab:
    def __init__(self, fname):
        self.fname = fname
        self.vocab = [line.strip() for line in open(fname, encoding='utf-8')]

    def get_metadata(self):
        return {
            'pad_id': 0,
            'unk_id': 100,
            'bos_id': 101,
            'eos_id': 102,
            'subword_prefix': '##',
        }

    def get_tokens(self):
        for text in self.vocab:
            yield text.encode('utf-8'), 0.0, TokenType.NORMAL

    def __repr__(self) -> str:
        return f'<SimpleVocab with {len(self.vocab)} tokens>'

# copied from llama.cpp
class SentencePieceVocab:
    def __init__(self, fname_tokenizer, fname_added_tokens=None):
        self.tokenizer = SentencePieceProcessor(str(fname_tokenizer))

        if fname_added_tokens is not None:
            added_tokens = json.load(open(fname_added_tokens, encoding='utf-8'))
        else:
            added_tokens = {}

        vocab_size       = self.tokenizer.vocab_size()
        new_tokens       = {id: piece for piece, id in added_tokens.items() if id >= vocab_size}
        expected_new_ids = list(range(vocab_size, vocab_size + len(new_tokens)))
        actual_new_ids   = sorted(new_tokens.keys())

        if expected_new_ids != actual_new_ids:
            raise ValueError(f'Expected new token IDs {expected_new_ids} to be sequential; got {actual_new_ids}')

        # Token pieces that were added to the base vocabulary.
        self.added_tokens_dict  = added_tokens
        self.added_tokens_list  = [new_tokens[id] for id in actual_new_ids]
        self.vocab_size_base    = vocab_size
        self.vocab_size         = self.vocab_size_base + len(self.added_tokens_list)
        self.fname_tokenizer    = fname_tokenizer
        self.fname_added_tokens = fname_added_tokens

    def get_metadata(self):
        return {
            'pad_id': self.tokenizer.pad_id(),
            'unk_id': self.tokenizer.unk_id(),
            'bos_id': self.tokenizer.bos_id(),
            'eos_id': self.tokenizer.eos_id(),
            'subword_prefix': '▁',
        }

    def regular_tokens(self):
        for i in range(self.tokenizer.vocab_size()):
            piece = self.tokenizer.id_to_piece(i)
            text = piece.encode('utf-8')
            score = self.tokenizer.get_score(i)

            toktype = TokenType.NORMAL
            if self.tokenizer.is_unknown(i):
                toktype = TokenType.UNKNOWN
            if self.tokenizer.is_control(i):
                toktype = TokenType.CONTROL

            # NOTE: I think added_tokens are user defined.
            # ref: https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto
            # if tokenizer.is_user_defined(i): toktype = TokenType.USER_DEFINED

            if self.tokenizer.is_unused(i):
                toktype = TokenType.UNUSED
            if self.tokenizer.is_byte(i):
                toktype = TokenType.BYTE

            yield text, score, toktype

    def added_tokens(self):
        for text in self.added_tokens_list:
            yield text.encode('utf-8'), -1000.0, TokenType.USER_DEFINED

    def get_tokens(self):
        yield from self.regular_tokens()
        yield from self.added_tokens()

    def __repr__(self) -> str:
        return f'<SentencePieceVocab with {self.vocab_size_base} base tokens and {len(self.added_tokens_list)} added tokens>'

# script usage
if __name__ == '__main__':
    # primay usage
    if len(sys.argv) < 2:
        print('Usage: convert-to-ggml.py model_dir [float-type=f16,f32]\n')
        sys.exit(1)

    # output in the same directory as the model
    model_dir = Path(sys.argv[1])
    float_type = sys.argv[2].lower() if len(sys.argv) > 2 else 'f16'

    # check model dir exists
    if not model_dir.exists():
        print(f'Directory {model_dir} does not exist.')
        sys.exit(1)

    # convert to ggml quantization type
    if float_type not in ['f16', 'f32']:
        print(f'Float type must be f16 or f32, got: {float_type}')
        sys.exit(1)
    else:
        qtype = GGMLQuantizationType[float_type.upper()]
        dtype0 = {'f16': torch.float16, 'f32': torch.float32}[float_type]

    # get output file name
    fname_out = model_dir / f'ggml-model-{float_type}.gguf'

    # heck if the directory existsc
    if not model_dir.exists():
        print(f'Directory {model_dir} does not exist.')

    # load model config
    with open(model_dir / 'config.json', 'r', encoding='utf-8') as f:
        hparams = json.load(f)

    # load tokenizer config
    with open(model_dir / 'tokenizer_config.json', 'r', encoding='utf-8') as f:
        tokenizer_meta = json.load(f)

    # load vocab
    vocab_path = model_dir / 'vocab.txt'
    bpe_path = model_dir / 'sentencepiece.bpe.model'
    if vocab_path.exists():
        vocab = SimpleVocab(vocab_path)
    elif bpe_path.exists():
        vocab = SentencePieceVocab(bpe_path)
    else:
        raise ValueError(f'No vocab file found in {model_dir}. Looked for: vocab.txt and sentencepiece.bpe.model')

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)

    # print model
    hparam_keys = [
        'vocab_size', 'max_position_embeddings', 'hidden_size', 'intermediate_size',
        'num_attention_heads', 'num_hidden_layers', 'layer_norm_eps'
    ]
    print('PARAMS')
    for k in hparam_keys:
        print(f'{k:<24s} = {hparams[k]}')
    print()

    # start to write GGUF file
    gguf_writer = GGUFWriter(fname_out, 'bert')

    # write metadata
    gguf_writer.add_name('BERT')
    gguf_writer.add_description('GGML BERT model')
    gguf_writer.add_file_type(qtype)

    # write model params
    gguf_writer.add_uint32('vocab_size', hparams['vocab_size'])
    gguf_writer.add_uint32('max_position_embedding', hparams['max_position_embeddings'])
    gguf_writer.add_uint32('hidden_size', hparams['hidden_size'])
    gguf_writer.add_uint32('intermediate_size', hparams['intermediate_size'])
    gguf_writer.add_uint32('num_attention_heads', hparams['num_attention_heads'])
    gguf_writer.add_uint32('num_hidden_layers', hparams['num_hidden_layers'])
    gguf_writer.add_float32('layer_norm_eps', hparams['layer_norm_eps'])

    # write vocab params
    vocab_meta = vocab.get_metadata()
    gguf_writer.add_int32(KEY_PAD_ID, vocab_meta['pad_id'])
    gguf_writer.add_int32(KEY_UNK_ID, vocab_meta['unk_id'])
    gguf_writer.add_int32(KEY_BOS_ID, vocab_meta['bos_id'])
    gguf_writer.add_int32(KEY_EOS_ID, vocab_meta['eos_id'])
    gguf_writer.add_string(KEY_SUBWORD_PREFIX, vocab_meta['subword_prefix'])

    # write vocab tokens
    token_list, score_list, type_list = zip(*vocab.get_tokens())
    gguf_writer.add_token_list(token_list)
    gguf_writer.add_token_scores(score_list)
    gguf_writer.add_token_types(type_list)

    # write tensors
    print('TENSORS')
    for name, data in model.state_dict().items():
        # get correct dtype
        if 'LayerNorm' in name or 'bias' in name:
            dtype = torch.float32
        else:
            dtype = dtype0

        # print info
        shape_str = str(list(data.shape))
        print(f'{name:64s} = {shape_str:16s} {data.dtype} → {dtype}')

        # do conversion
        data = data.to(dtype)

        # add to gguf output
        gguf_writer.add_tensor(name, data.numpy())

    # execute and close writer
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    # print success
    print(f'GGML model written to {fname_out}')
