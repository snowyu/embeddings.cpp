import os
import ctypes
import numpy as np

LIB_DIR = os.path.dirname(__file__)
LIB_PATH = os.path.join(LIB_DIR, '../build/libbert.so')

class BertModel:
    def __init__(self, fname):
        # set up ctypes for library
        self.lib = ctypes.cdll.LoadLibrary(LIB_PATH)

        self.lib.bert_load_from_file.restype = ctypes.c_void_p
        self.lib.bert_load_from_file.argtypes = [
            ctypes.c_char_p, # const char * fname
            ctypes.c_bool,   # bool use_cpu
        ]

        self.lib.bert_n_embd.restype = ctypes.c_int32
        self.lib.bert_n_embd.argtypes = [ctypes.c_void_p]
        
        self.lib.bert_free.argtypes = [ctypes.c_void_p]

        self.lib.bert_encode_batch_c.argtypes = [
            ctypes.c_void_p,                 # struct bert_ctx * ctx
            ctypes.POINTER(ctypes.c_char_p), # const char ** texts
            ctypes.POINTER(ctypes.c_float),  # float * embeddings
            ctypes.c_int32,                  # int32_t n_inputs
            ctypes.c_int32,                  # int32_t n_threads
        ]

        # load model from file and get embedding size
        self.ctx = self.lib.bert_load_from_file(fname.encode('utf-8'), True)
        self.n_embd = self.lib.bert_n_embd(self.ctx)

    def __del__(self):
        self.lib.bert_free(self.ctx)

    def encode(self, batch, n_threads=6, batch_size=16):
        # handle singleton case
        if isinstance(batch, str):
            batch = [batch]
            squeeze = True
        else:
            squeeze = False
        n_input = len(batch)

        # create embedding memory
        embed = np.zeros((n_input, self.n_embd), dtype=np.float32)
        embed_p = embed.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # create batch strings
        strings = (ctypes.c_char_p * n_input)()
        for j, s in enumerate(batch):
            strings[j] = s.encode('utf-8')

        # call bert.cpp function
        self.lib.bert_encode_batch_c(self.ctx, strings, embed_p, n_input, n_threads)
        return embed[0] if squeeze else embed
