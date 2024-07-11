from transformers import BertConfig, BertModel
from transformers import BertModel
import torch
from transformers import BertTokenizer
from transformers import AutoTokenizer

config = BertConfig()
model = BertModel(config)

model = BertModel.from_pretrained("bert-base-cased")

model.save_pretrained("directory_on_my_computer")

sequences = ["Hello!", "Cool.", "Nice!"]

encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]

model_inputs = torch.tensor(encoded_sequences)

output = model(model_inputs)

tokenized_text = "Jim Henson was a puppeteer".split()
print(tokenized_text)

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
tokenizer("Using a Transformer network is simple")
tokenizer.save_pretrained("directory_on_my_computer")

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)