import torch
from transformers.configuration_bart import BartConfig
from transformers.modeling_bart_edit import BartModelEdit, BartForConditionalGeneration, BartForSequenceClassification


config = BartConfig.from_pretrained('facebook/bart-large-cnn')
config.decoder_attention_heads = 12
config.decoder_layers = 6
config.decoder_ffn_dim = 768 * 4
config.d_model = 768
config.encoder_attention_heads = 12
config.encoder_layers = 6
config.encoder_ffn_dim = 768 * 4

bart = BartModelEdit(config)
bart_generation = BartForConditionalGeneration(config)
bart_classification = BartForSequenceClassification(config)

print(bart)
print('\n============================================================\n')
print(bart_generation)
print('\n============================================================\n')
print(bart_classification)