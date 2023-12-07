from transformers import AutoTokenizer, AutoModel
import tiktoken

tokenizer_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
# tokenizer_name = "jinaai/jina-embeddings-v2-base-en"
# tokenizer_name = "mymusise/CPM-GPT2"
# tokenizer_name = "gpt2"
# tokenizer_name = "bert-base-chinese"
# tokenizer_name = "BAAI/llm-embedder"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

inps = [
        "Hell12o,, world! complexness123", 
        "，大12家好gpt，我是GPT。你好。中國龍", 
        "你好，世界！", 
        "こんにちは、世界！", 
        "syömme \t  täällä tänään",
        "🙂🙂🙂😒😒😒🍍🍍🍑😗⚜️🕕⛄☃️",
        "1231 2431431",
    ]

print("Using tokenizer:", tokenizer_name)
for inp in inps:
    oup = tokenizer(inp, return_tensors="pt").input_ids[0].tolist()
    print(f"{oup} is {tokenizer.decode(oup)}")
    for token in oup:
        print(f"{token} <--> {tokenizer.decode([token])}")
    print("\n\n")
    # print(f"{oup} is {tokenizer.decode(oup)}")

    # print(f"{inp} is tokenized as {oup}")
    # for token in oup:
        # print(f"{token} is {tokenizer.decode([token])}")

