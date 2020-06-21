from build_edited_bart_model import init_edited_bart_model
from transformers import BartTokenizer, BartConfig

model = init_edited_bart_model('gen')
model.eval()
print(model)
tokenizer = BartTokenizer.from_pretrained('bart-large')
ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
inputs = tokenizer.batch_encode_plus([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')
# Generate Summary
summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])



TXT = "My friends are <mask> but they eat too many carbs."
input_ids = tokenizer.batch_encode_plus([TXT], return_tensors='pt')['input_ids']
print(input_ids)
logits = model(input_ids)[0]
print(logits)
masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
print(masked_index)
probs = logits[0, masked_index].softmax(dim=0)
values, predictions = probs.topk(5)
print(tokenizer.decode(predictions).split())