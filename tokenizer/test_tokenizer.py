from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer/mobilecap_tokenizer.json")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

encoded = tokenizer.encode("A cat sitting on a big mac cheese couch.")
print(f"Tokens: {encoded}")
print(f"Decoded: {tokenizer.decode(encoded)}")
# Should look like: [CLS] A cat sitting on a couch [SEP]