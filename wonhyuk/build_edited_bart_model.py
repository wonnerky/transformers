import torch
from transformers.configuration_bart import BartConfig
from transformers.modeling_bart_edit import BartModelEdit

def edited_Bart_Model():
    bart_trained = torch.hub.load('pytorch/fairseq', 'bart.base')
    bart_trained.train()
    trained_decoder = bart_trained.model.decoder

    config = BartConfig.from_pretrained('facebook/bart-large-cnn')
    config.decoder_attention_heads = 12
    config.decoder_layers = 6
    config.decoder_ffn_dim = 768 * 4
    config.d_model = 768
    config.encoder_attention_heads = 12
    config.encoder_layers = 6
    config.encoder_ffn_dim = 768 * 4
    edited_bart_model = BartModelEdit(config)
    edited_bart_model.train()

    # Print Model Parameters
    # print('edited Model : \n', [i for i in edited_bart_model.decoder.layers[0].self_attn.k_proj.parameters()])
    # print('===================================================================')
    # print('===================================================================')
    # print('pretrained Model : \n', [i for i in trained_decoder.layers[0].self_attn.k_proj.parameters()])

    # Check parameters between pre_trained model and transfer_model
    # if torch.equal(next(edited_bart_model.decoder.layers[0].self_attn.k_proj.parameters()).data, next(trained_decoder.layers[0].self_attn.k_proj.parameters()).data):
    #     print('Yes!!')
    # else:
    #     print('No!!!')

    pretrained_dict = trained_decoder.state_dict()
    model_dict = edited_bart_model.state_dict()
    pretrained_dict = {'decoder.' + k: v for k, v in pretrained_dict.items() if 'decoder.' + k in model_dict}
    del(pretrained_dict['decoder.embed_tokens.weight'])
    model_dict.update(pretrained_dict)
    edited_bart_model.load_state_dict(model_dict)

    # Check models after transfer
    # print('edited Model : \n', [i for i in edited_bart_model.decoder.layers[0].self_attn.k_proj.parameters()])
    # print('===================================================================')
    # print('===================================================================')
    # print('pretrained Model : \n', [i for i in trained_decoder.layers[0].self_attn.k_proj.parameters()])
    # if torch.equal(next(edited_bart_model.decoder.layers[0].self_attn.k_proj.parameters()).data, next(trained_decoder.layers[0].self_attn.k_proj.parameters()).data):
    #     print('Yes!!')
    # else:
    #     print('No!!!')

    return edited_bart_model





# Model weight 초기화
# print(k)
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
# b.apply(init_weights)

