from transformers import AutoTokenizer
from torch import nn
from transformers import EsmModel
from util import printl


def get_tokenizer(configs):
    if 'facebook/esm2' in configs.encoder_name:
        tokenizer = AutoTokenizer.from_pretrained(configs.encoder_name)
    else:
        raise ValueError("Wrong tokenizer specified.")
    return tokenizer


def get_model(configs):
    if 'facebook/esm2' in configs.encoder_name:
        encoder = Encoder(configs=configs)
    else:
        raise ValueError("Wrong model specified")
    return encoder


def get_esm(configs, log_path):
    model = EsmModel.from_pretrained(configs.encoder_name)
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    if configs.training_mode == "frozen":
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        printl("All model parameters have been frozen.", log_path=log_path)
    elif configs.training_mode == "finetune":
        # Allow the parameters of the last transformer block to be updated during fine-tuning
        for param in model.encoder.layer[configs.train_settings.fine_tune_layer:].parameters():
            param.requires_grad = True
        printl(f"Parameters in the last {len(model.encoder.layer) - configs.train_settings.fine_tune_layer} layers are trainable.", log_path=log_path)
        for param in model.pooler.parameters():
            param.requires_grad = False
        printl("Pooling layer parameters have been frozen.", log_path=log_path)
    return model

class Encoder(nn.Module):
    def __init__(self, configs, log_path):
        super().__init__()
        self.model = get_esm(configs, log_path)
        # self.pooling_layer = nn.AdaptiveAvgPool2d((None, 1))
        # self.pooling_layer = nn.AdaptiveAvgPool1d(1)

    # def forward(self, encoded_sequence, id, id_frags_list, seq_frag_tuple, pos_neg, warm_starting):
    #     classification_head = None
    #     motif_logits = None
    #     projection_head = None
    #     if self.skip_esm:
    #         emb_pro_list = []
    #         for i in id:
    #             emb_pro_list.append(self.protein_embeddings[i])
    #
    #         emb_pro = torch.stack(emb_pro_list, dim=0)
    #     else:
    #         # print(encoded_sequence['attention_mask'].shape)
    #         # print(encoded_sequence['attention_mask'])
    #         # print(encoded_sequence['input_ids'])
    #         # exit(0)
    #         # 这
    #         features = self.model(input_ids=encoded_sequence['input_ids'],
    #                               attention_mask=encoded_sequence['attention_mask'])
    #         # print(features)
    #         last_hidden_state = remove_s_e_token(features.last_hidden_state,
    #                                              encoded_sequence['attention_mask'])  # [batch, maxlen-2, dim]
    #
    #         if not self.combine:  # only need type_head if combine is False
    #             emb_pro = self.get_pro_emb(id, id_frags_list, seq_frag_tuple, last_hidden_state, self.overlap)
    #
    #     if self.apply_supcon:
    #         if not warm_starting:
    #             motif_logits = self.ParallelDecoders(last_hidden_state)
    #             if self.combine:
    #                 classification_head = self.get_pro_class(self.predict_max, id, id_frags_list, seq_frag_tuple,
    #                                                          motif_logits, self.overlap)
    #             else:
    #                 classification_head = self.type_head(emb_pro)  # [sample, num_class]
    #
    #             if pos_neg is not None:
    #                 """CASE B, when this if condition is skipped, CASE A"""
    #                 projection_head = self.projection_head(self.reorganize_emb_pro(emb_pro))
    #         else:
    #             """CASE C"""
    #             projection_head = self.projection_head(self.reorganize_emb_pro(emb_pro))
    #     else:
    #         """CASE D"""
    #         """这"""
    #         motif_logits = self.ParallelDecoders(
    #             last_hidden_state)  # list no shape # last_hidden_state=[batch, maxlen-2, dim]
    #         if self.combine:
    #             classification_head = self.get_pro_class(self.predict_max, id, id_frags_list, seq_frag_tuple,
    #                                                      motif_logits, self.overlap)
    #             # print('classification_head', classification_head.shape)
    #             if self.combine_DNN:
    #                 classification_head = self.DNN_head(classification_head)
    #                 # print('classification_head', classification_head.shape)
    #         else:
    #             # print('emb_pro', emb_pro.shape)
    #             classification_head = self.type_head(emb_pro)  # [sample, num_class]
    #             # print('classification_head', classification_head.shape)
    #
    #     # print(motif_logits[0,0,:])
    #     # print(motif_logits.shape)
    #     # print(motif_logits[0,1,:])
    #     # maxvalues,_ = torch.max(motif_logits[0], dim=-1, keepdim=True)
    #     # print(maxvalues)
    #     return classification_head, motif_logits, projection_head