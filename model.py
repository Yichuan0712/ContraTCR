from transformers import AutoTokenizer
from torch import nn

def get_tokenizer(configs):
    if 'facebook/esm2' in configs.encoder_name:
        tokenizer = AutoTokenizer.from_pretrained(configs.encoder_name)
    else:
        raise ValueError("Wrong tokenizer specified.")
    return tokenizer

def get_model(configs):
    if 'facebook/esm2' in configs.encoder_name:
        encoder = Encoder(model_name=configs.encoder.model_name,
                          model_type=configs.encoder.model_type,
                          configs=configs
                          )
    else:
        raise ValueError("Wrong model specified")
    return encoder

class Encoder(nn.Module):
    def __init__(self, configs, model_name='facebook/esm2_t33_650M_UR50D', model_type='esm_v2'):
        super().__init__()
        self.model_type = model_type
        if model_type == 'esm_v2':
            self.model = prepare_esm_model(model_name, configs)
        # self.pooling_layer = nn.AdaptiveAvgPool2d((None, 1))
        self.combine = configs.decoder.combine
        self.combine_DNN = configs.decoder.combine_DNN
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        if configs.decoder.type == "linear":
            self.ParallelDecoders = ParallelLinearDecoders(input_size=self.model.config.hidden_size,
                                                           output_sizes=[1] * configs.encoder.num_classes)
        elif configs.decoder.type == "cnn":
            self.ParallelDecoders = ParallelCNNDecoders(input_size=self.model.config.hidden_size,
                                                        output_sizes=[1] * configs.encoder.num_classes,
                                                        out_channels=configs.decoder.cnn_channel,
                                                        kernel_size=configs.decoder.cnn_kernel,
                                                        droprate=configs.decoder.droprate,
                                                        )
        elif configs.decoder.type == "cnn-linear":
            self.ParallelDecoders = ParallelCNN_Linear_Decoders(input_size=self.model.config.hidden_size,
                                                                cnn_output_sizes=[1] * 2,  # cnn
                                                                linear_output_sizes=[1] * (
                                                                            configs.encoder.num_classes - 2),
                                                                out_channels=configs.decoder.cnn_channel,
                                                                kernel_size=configs.decoder.cnn_kernel,
                                                                droprate=configs.decoder.droprate,
                                                                )
        if not self.combine:  # only need type_head if combine is False
            self.type_head = nn.Linear(self.model.embeddings.position_embeddings.embedding_dim,
                                       configs.encoder.num_classes)
        if self.combine == False and self.combine_DNN == True:
            print('combine and combine_DNN must be both True')
            raise 'combine and combine_DNN must be both True'
        if self.combine and self.combine_DNN:
            self.DNN_head = nn.Linear(configs.encoder.num_classes,
                                       configs.encoder.num_classes)

        self.overlap = configs.encoder.frag_overlap

        # SupCon
        self.apply_supcon = configs.supcon.apply
        if self.apply_supcon:
            self.projection_head = LayerNormNet(configs)
            self.n_pos = configs.supcon.n_pos
            self.n_neg = configs.supcon.n_neg
            self.batch_size = configs.train_settings.batch_size
            # new code

        self.skip_esm = configs.supcon.skip_esm
        if self.skip_esm:
            self.protein_embeddings = torch.load("5283_esm2_t33_650M_UR50D.pt")

        self.predict_max = configs.train_settings.predict_max


    def get_pro_emb(self, id, id_frags_list, seq_frag_tuple, emb_frags, overlap):
        # print(seq_frag_tuple)
        # print('emb_frag', emb_frags.shape)
        emb_pro_list = []
        for id_protein in id:
            ind_frag = 0
            id_frag = id_protein + "@" + str(ind_frag)
            while id_frag in id_frags_list:
                ind = id_frags_list.index(id_frag)
                emb_frag = emb_frags[ind]  # [maxlen-2, dim]
                seq_frag = seq_frag_tuple[ind]
                l = len(seq_frag)
                if ind_frag == 0:
                    emb_pro = emb_frag[:l]
                else:
                    overlap_emb = (emb_pro[-overlap:] + emb_frag[:overlap]) / 2
                    emb_pro = torch.concatenate((emb_pro[:-overlap], overlap_emb, emb_frag[overlap:l]), axis=0)
                ind_frag += 1
                id_frag = id_protein + "@" + str(ind_frag)
            # print('-before mean', emb_pro.shape)
            emb_pro = torch.mean(emb_pro, dim=0)
            # print('-after mean', emb_pro.shape)
            emb_pro_list.append(emb_pro)

        emb_pro_list = torch.stack(emb_pro_list, dim=0)
        return emb_pro_list

    def get_pro_class(self, predict_max, id, id_frags_list, seq_frag_tuple, motif_logits, overlap):
        # motif_logits_max, _ = torch.max(motif_logits, dim=-1, keepdim=True).squeeze(-1) #should be [batch,num_class]
        # print(motif_logits_max)
        motif_pro_list = []
        for id_protein in id:
            ind_frag = 0
            id_frag = id_protein + "@" + str(ind_frag)
            while id_frag in id_frags_list:
                ind = id_frags_list.index(id_frag)
                motif_logit = motif_logits[ind]  # [num_class,max_len]
                seq_frag = seq_frag_tuple[ind]
                l = len(seq_frag)
                if ind_frag == 0:
                    motif_pro = motif_logit[:, :l]  # [num_class,length]
                else:
                    overlap_motif = (motif_pro[:, -overlap:] + motif_logit[:, :overlap]) / 2
                    motif_pro = torch.concatenate((motif_pro[:, :-overlap], overlap_motif, motif_logit[:, overlap:l]),
                                                  axis=-1)
                ind_frag += 1
                id_frag = id_protein + "@" + str(ind_frag)

            if predict_max:
                # print('-before max', motif_pro.shape)  # should be [num_class,length]
                motif_pro, _ = torch.max(motif_pro, dim=-1)
                # print('-after max', motif_pro.shape)  # should be [num_class]
            else:
                # print('-before mean', motif_pro.shape)  # should be [num_class,length]
                # motif_pro = torch.mean(motif_pro, dim=-1)
                motif_pro = torch.mean(motif_pro.clone(), dim=-1)  # yichuan 0605
                # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
                # 只在特定版本pytorch上出现此错误
                # print('-after mean', motif_pro.shape)  # should be [num_class]

            motif_pro_list.append(motif_pro)  # [batch,num_class]

        motif_pro_list = torch.stack(motif_pro_list, dim=0)
        return motif_pro_list

    def reorganize_emb_pro(self, emb_pro):
        n_batch = int(emb_pro.shape[0] / (1 + self.n_pos + self.n_neg))
        bch_anchors, bch_positives, bch_negatives = torch.split(emb_pro,
                                                                [n_batch, n_batch * self.n_pos, n_batch * self.n_neg],
                                                                dim=0)
        emb_pro_ = []
        for i in range(n_batch):
            anchor = bch_anchors[i].unsqueeze(0)
            positive = bch_positives[(i * self.n_pos):(i * self.n_pos + self.n_pos)]
            negative = bch_negatives[(i * self.n_neg):(i * self.n_neg + self.n_neg)]
            triple = torch.cat((anchor, positive, negative), dim=0)
            emb_pro_.append(triple)
        emb_pro_ = torch.stack(emb_pro_, dim=0)
        return emb_pro_

    def forward(self, encoded_sequence, id, id_frags_list, seq_frag_tuple, pos_neg, warm_starting):
        """
        Batch is built before forward(), in train_loop()
        Batch is either (anchor) or (anchor+pos+neg)

        if apply supcon:
            if not warming starting:
                if pos_neg is None: ------------------> batch should be built as (anchor) in train_loop()
                    "CASE A"
                    get motif_logits from batch
                    get classification_head from batch
                else: --------------------------------> batch should be built as (anchor+pos+neg) in train_loop()
                    "CASE B"
                    get motif_logits from batch
                    get classification_head from batch
                    get projection_head from batch
            else: ------------------------------------> batch should be built as (anchor+pos+neg) in train_loop()
                "CASE C"
                get projection_head from batch
        else: ----------------------------------------> batch should be built as (anchor) in train_loop()
            "CASE D"
            get motif_logits from batch
            get classification_head from batch
        """
        classification_head = None
        motif_logits = None
        projection_head = None
        if self.skip_esm:
            emb_pro_list = []
            for i in id:
                emb_pro_list.append(self.protein_embeddings[i])

            emb_pro = torch.stack(emb_pro_list, dim=0)
        else:
            # print(encoded_sequence['attention_mask'].shape)
            # print(encoded_sequence['attention_mask'])
            # print(encoded_sequence['input_ids'])
            # exit(0)
            # 这
            features = self.model(input_ids=encoded_sequence['input_ids'],
                                  attention_mask=encoded_sequence['attention_mask'])
            # print(features)
            last_hidden_state = remove_s_e_token(features.last_hidden_state,
                                                 encoded_sequence['attention_mask'])  # [batch, maxlen-2, dim]

            if not self.combine:  # only need type_head if combine is False
                emb_pro = self.get_pro_emb(id, id_frags_list, seq_frag_tuple, last_hidden_state, self.overlap)

        if self.apply_supcon:
            if not warm_starting:
                motif_logits = self.ParallelDecoders(last_hidden_state)
                if self.combine:
                    classification_head = self.get_pro_class(self.predict_max, id, id_frags_list, seq_frag_tuple,
                                                             motif_logits, self.overlap)
                else:
                    classification_head = self.type_head(emb_pro)  # [sample, num_class]

                if pos_neg is not None:
                    """CASE B, when this if condition is skipped, CASE A"""
                    projection_head = self.projection_head(self.reorganize_emb_pro(emb_pro))
            else:
                """CASE C"""
                projection_head = self.projection_head(self.reorganize_emb_pro(emb_pro))
        else:
            """CASE D"""
            """这"""
            motif_logits = self.ParallelDecoders(
                last_hidden_state)  # list no shape # last_hidden_state=[batch, maxlen-2, dim]
            if self.combine:
                classification_head = self.get_pro_class(self.predict_max, id, id_frags_list, seq_frag_tuple,
                                                         motif_logits, self.overlap)
                # print('classification_head', classification_head.shape)
                if self.combine_DNN:
                    classification_head = self.DNN_head(classification_head)
                    # print('classification_head', classification_head.shape)
            else:
                # print('emb_pro', emb_pro.shape)
                classification_head = self.type_head(emb_pro)  # [sample, num_class]
                # print('classification_head', classification_head.shape)

        # print(motif_logits[0,0,:])
        # print(motif_logits.shape)
        # print(motif_logits[0,1,:])
        # maxvalues,_ = torch.max(motif_logits[0], dim=-1, keepdim=True)
        # print(maxvalues)
        return classification_head, motif_logits, projection_head