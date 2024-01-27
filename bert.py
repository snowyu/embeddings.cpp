import os
import ctypes
import numpy as np

LIB_DIR = os.path.dirname(__file__)
LIB_PATH = os.path.join(LIB_DIR, 'build/libbert.so')

def increment_pointer(p, d):
    t = type(p)._type_
    v = ctypes.cast(p, ctypes.c_void_p)
    v.value += d * ctypes.sizeof(t)
    return ctypes.cast(v, ctypes.POINTER(t))

class BertModel:
    def __init__(self, fname, batch_size=32, use_cpu=False):
        # set up ctypes for library
        self.lib = ctypes.cdll.LoadLibrary(LIB_PATH)

        self.lib.bert_load_from_file.restype = ctypes.c_void_p
        self.lib.bert_load_from_file.argtypes = [
            ctypes.c_char_p, # const char * fname
            ctypes.c_int32,  # int batch_size
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
        self.ctx = self.lib.bert_load_from_file(fname.encode('utf-8'), batch_size, use_cpu)
        self.n_embd = self.lib.bert_n_embd(self.ctx)
        self.batch_size = batch_size

    def __del__(self):
        self.lib.bert_free(self.ctx)

    def encode_batch(self, batch, embed_p=None, n_threads=8):
        # create embedding memory
        n_input = len(batch)
        if embed_p is None:
            embed = np.zeros((n_input, self.n_embd), dtype=np.float32)
            embed_p = embed.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            embed = None

        # create batch strings
        strings = (ctypes.c_char_p * n_input)()
        for j, s in enumerate(batch):
            strings[j] = s.encode('utf-8')

        # call bert.cpp function
        self.lib.bert_encode_batch_c(self.ctx, strings, embed_p, n_input, n_threads)

        # return if it wasn't inplace
        if embed is not None:
            return embed

    def encode(self, text):
        # handle singleton case
        if isinstance(text, str):
            text = [text]
            squeeze = True
        else:
            squeeze = False
        n_input = len(text)

        # create embedding memory
        embed = np.zeros((n_input, self.n_embd), dtype=np.float32)
        embed_p = embed.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # loop over batches
        for i in range(0, n_input, self.batch_size):
            j = min(i + self.batch_size, n_input)
            batch = text[i:j]
            batch_p = increment_pointer(embed_p, i * self.n_embd)
            self.encode_batch(batch, embed_p=batch_p)

        # return squeezed maybe
        return embed[0] if squeeze else embed
