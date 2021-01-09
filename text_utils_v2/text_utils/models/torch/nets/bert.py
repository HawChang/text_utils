import json
import logging
import math
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

#def get_device(tensor):
#    device_name = 'cpu'
#    try:
#        device_id = tensor.get_device()
#        device_name = 'cuda:{}'.format(device_id) if device_id >= 0 else 'cpu'
#    except RuntimeError as e:
#        pass
#    #logging.warning("get device_name: {}".format(device_name))
#    return device_name

def swish(x):
    return x * torch.sigmoid(x)

def gelu(x):
    """ 
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "mish": mish}

class BertConfig(object):
    def __init__(
        self,
        vocab_size=None,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        **kwargs
    ):

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        for attr, value in kwargs.items():
            setattr(self, attr, value)
            logging.info("extra bert config: {} = {}".format(attr, value))


class BertLayerNorm(nn.Module):
    """LayerNorm层, 见Transformer(一), 讲编码器(encoder)的第3部分"""
    def __init__(self, hidden_size, eps=1e-12, conditional=False):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.conditional = conditional
        if conditional == True:
            #说明是条件 ln
            self.weight_dense = nn.Linear(2 * hidden_size, hidden_size, bias=False)
            self.weight_dense.weight.data.uniform_(0, 0)
            self.bias_dense = nn.Linear(2 * hidden_size, hidden_size, bias=False)
            self.bias_dense.weight.data.uniform_(0, 0)

    def forward(self, x):
        if self.conditional == False:
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.gamma * x + self.beta
        else :
            inputs = x[0]
            cond = x[1]
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(dim=1)

            weight = self.gamma + self.weight_dense(cond)
            bias = self.beta + self.bias_dense(cond)
            u = inputs.mean(-1, keepdim=True)
            s = (inputs - u).pow(2).mean(-1, keepdim=True)
            x = (inputs - u) / torch.sqrt(s + self.variance_epsilon)

            return weight * x + bias



class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None):
        # input_ids shape: [batch_size, seq_length]

        logging.debug("input_shape: {}".format(input_ids.shape))

        input_shape = input_ids.size()

        seq_length = input_shape[1]
        device = input_ids.device
        # 如果没有传position_ids 则默认为序列[0,seq_length)
        if position_ids is None:
            # cur shape: [seq_length]
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            # cur shape: [batch_size, seq_length]
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        # position_ids shape: [batch_size, seq_length]
        logging.debug("position_ids shape: {}".format(position_ids.shape))

        # 如果没有传入token_type_ids 默认为同一句话 因此id全为0
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # inputs_embeds shape : [batch_size, seq_length, hidden_size]
        inputs_embeds = self.word_embeddings(input_ids)
        logging.debug("inputs_embeds shape: {}".format(inputs_embeds.shape))

        # position_embeddings shape : [batch_size, seq_length, hidden_size]
        position_embeddings = self.position_embeddings(position_ids)
        logging.debug("position_embeddings shape: {}".format(position_embeddings.shape))

        # token_type_embeddings shape : [batch_size, seq_length, hidden_size]
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        logging.debug("token_type_embeddings shape: {}".format(token_type_embeddings.shape))

        # 全加起来
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        logging.debug("embeddings shape: {}".format(embeddings.shape))

        # layernorm + dropout
        embeddings = self.LayerNorm(embeddings)
        logging.debug("embeddings shape: {}".format(embeddings.shape))

        embeddings = self.dropout(embeddings)
        logging.debug("embeddings shape: {}".format(embeddings.shape))
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        # 多头注意力的头数
        self.num_attention_heads = config.num_attention_heads
        logging.debug("num_attention_heads: {}".format(self.num_attention_heads))
        # 每头的注意力维数 取整
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        logging.debug("attention_head_size: {}".format(self.attention_head_size))
        # 多头注意力总共的维数
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # x shape: [batch_size, seq_length, self.all_head_size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # x shape: [batch_size, seq_length, self.num_attention_heads, self.attention_head_size]
        x = x.view(*new_x_shape)

        ## 最后xshape (batch_size, num_attention_heads, seq_len, head_size)
        # x shape: [batch_size, self.num_attention_heads, seq_length, self.attention_head_size]
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False
    ):
        # hidden_states shape: [batch_size, seq_length, config.hidden_size]
        # attention_mask shape: [batch_size, 1, seq_length, seq_length]
        # self attention q,k,v为同一个值

        # 三个shape相同: [batch_size, seq_length, self.all_head_size]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # 三个shape相同: [batch_size, self.num_attention_heads, seq_length, self.attention_head_size]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores shape: [batch_size, self.num_attention_heads, seq_length, seq_length]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # attention_mask: 被mask的部分为 -10000.0，未mask的部分未0
        # attention_mask会直接与raw_attention_score相加 然后过softmax操作
        # 则被mask的部分会极其小 相当于被忽略
        # attention_scores shape: [batch_size, self.num_attention_heads, seq_length, seq_length]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # attention_probs shape: [batch_size, self.num_attention_heads, seq_length, seq_length]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # 注意力加权 torch.dot()
        # context_layer: [batch_size, self.num_attention_heads, seq_length, self.attention_head_size]
        context_layer = torch.matmul(attention_probs, value_layer)

        # 把加权后的V reshape, 得到[batch_size, length, embedding_dimension]
        # context_layer: [batch_size, seq_length, self.num_attention_heads, self.attention_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer: [batch_size, seq_length, self.all_head_size]
        context_layer = context_layer.view(*new_context_layer_shape)

        # 得到输出
        if output_attentions:
            return context_layer, attention_probs
        return context_layer, None


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # hidden_states shape: [batch_size, seq_length, all_head_size(config.hidden_size)]
        # input_tensor shape: [batch_size, seq_length, config.hidden_size]

        # hidden_states shape: [batch_size, seq_length, config.hidden_size]
        # projection
        hidden_states = self.dense(hidden_states)
        # dropout
        hidden_states = self.dropout(hidden_states)
        # add + layer norm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False
    ):
        # hidden_states shape: [batch_size, seq_length, config.hidden_size]
        # attention_mask shape: [batch_size, 1, seq_length, seq_length]

        # multi head attention
        # self_outputs: [batch_size, seq_length, all_head_size]。其中，all_head_size==config.hidden_size
        # attention_matrix shape: [batch_size, self.num_attention_heads, seq_length, seq_length]
        self_outputs, attention_metrix = self.self(hidden_states, attention_mask, output_attentions=output_attentions)

        # add + layer norm
        # attention_output shape: [batch_size, seq_length, config.hidden_size]
        attention_output = self.output(self_outputs, hidden_states)

        return attention_output, attention_metrix


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] ## relu 

    def forward(self, hidden_states):
        # hidden_states shape: [batch_size, seq_length, config.hidden_size]
        # fc
        # hidden_states shape: [batch_size, seq_length, config.intermediate_size]
        hidden_states = self.dense(hidden_states)
        # 激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # hidden_states shape: [batch_size, seq_length, config.intermediate_size]
        # input_tensor shape: [batch_size, seq_length, config.hidden_size]

        # hidden_states shape: [batch_size, seq_length, config.hidden_size]
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # add + layer norm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False
    ):
        # hidden_states shape: [batch_size, seq_length, config.hidden_size]
        # attention_mask shape: [batch_size, 1, seq_length, seq_length]

        # multi head attention + add + layer norm
        # attention_output shape: [batch_size, seq_length, config.hidden_size]
        # attention_matrix shape: [batch_size, self.num_attention_heads, seq_length, seq_length]
        attention_output, attention_matrix = self.attention(hidden_states, attention_mask, output_attentions=output_attentions)

        # feed forward
        # intermediate_output shape: [batch_size, seq_length, config.intermediate_size]
        intermediate_output = self.intermediate(attention_output)

        # add + layer norm
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output, attention_matrix


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_all_encoded_layers=True,
        output_attentions=False
    ):
        all_encoder_layers = []
        all_attention_matrices = []
        for i, layer_module in enumerate(self.layer):

            layer_output, attention_matrix = layer_module(
                hidden_states, attention_mask, output_attentions=output_attentions
            )
            hidden_states = layer_output
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                all_attention_matrices.append(attention_matrix)

        # 如果只输出最后一层 则这里加上最后一层
        # 如果输出所有层 则循环里就添加了 这里不需要
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            all_attention_matrices.append(attention_matrix)

        return all_encoder_layers, all_attention_matrices


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # hidden_states shape: [batch_size, seq_length, config.hidden_size]
        logging.debug("hidden_states shape: {}".format(hidden_states.shape))
        first_token_tensor = hidden_states[:, 0]
        # first_token_tensor shape: [batch_size, config.hidden_size]
        logging.debug("first_token_tensor shape: {}".format(first_token_tensor.shape))
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        # hidden_states shape: [batch_size, seq_length, config.hidden_size]
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        # hidden_states shape: [batch_size, seq_length, config.hidden_size]
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        logging.debug("hidden_states shape: {}".format(hidden_states.shape))

        # hidden_states shape: [batch_size, seq_length, config.hidden_size]
        hidden_states = self.transform(hidden_states)
        logging.debug("hidden_states shape: {}".format(hidden_states.shape))

        # hidden_states shape: [batch_size, seq_length, config.vocab_size]
        hidden_states = self.decoder(hidden_states)
        logging.debug("hidden_states shape: {}".format(hidden_states.shape))
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_path, keep_tokens=None, **kwargs):
        assert "vocab_size" in kwargs, \
                "parameter 'vocab_size' is required when load bert"

        bert_state_dict_path = os.path.join(pretrained_model_path,
                "bert-base-chinese-pytorch_model.bin")
        assert os.path.exists(bert_state_dict_path), \
                "cannot find state dict file: {}".format(bert_state_dict_path)

        bert_config_path = os.path.join(pretrained_model_path, 'bert_config.json')
        assert os.path.exists(bert_config_path), \
                "cannot find bert config file: {}".format(bert_config_path)

        with open(bert_config_path) as rf:
            bert_config_dict = dict(json.loads(rf.read()), **kwargs)
            logging.info("bert_config_dict: {}".format(bert_config_dict))
        bert_config_dict = BertConfig(**bert_config_dict)
        model = cls(bert_config_dict)

        pretrained_state_dict = torch.load(bert_state_dict_path)

        logging.debug("remove state dict: {}".format(
            [k for k in pretrained_state_dict.keys() \
                    if k[:4] != "bert" or "pooler" in k]))

        # 去除pooler和非bert相关的权重
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() \
                if k[:4] == "bert" and "pooler" not in k}

        # 预训练数据有的权值
        pretrained_state_dict_name_set = pretrained_state_dict.keys()
        # 模型需要的权重
        model_state_dict_name_set = model.state_dict().keys()

        logging.info("unused weight in pretrained file: {}".format(
            pretrained_state_dict_name_set - model_state_dict_name_set))
        logging.info("missing weight in pretrained file: {}".format(
            model_state_dict_name_set - pretrained_state_dict_name_set))

        if keep_tokens is not None:
            ## 说明精简词表了，embeedding层也要过滤下
            embedding_weight_name = "bert.embeddings.word_embeddings.weight"

            pretrained_state_dict[embedding_weight_name] = \
                    pretrained_state_dict[embedding_weight_name][keep_tokens]

        model.load_state_dict(pretrained_state_dict, strict=False)
        torch.cuda.empty_cache()
        logging.info("succeed loading model from {}".format(pretrained_model_path))

        return model

    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear)):
            # 初始线性映射层的参数为正态分布
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            # 初始化LayerNorm中的alpha为全1, beta为全0
            module.beta.data.zero_()
            module.gamma.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            # 初始化偏置为0
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.
    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762
    """

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(self.config)
        self.encoder = BertEncoder(self.config)
        self.pooler = BertPooler(self.config)

        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        output_all_encoded_layers=True,
        output_attentions=False
    ):

        # 0为pad id 这里是将pad mask掉
        # input ids shape: [batch_size, seq_length]
        logging.debug("input ids shape: {}".format(input_ids.shape))
        extended_attention_mask = (input_ids > 0).float()

        # 注意力矩阵mask: [batch_size, 1, 1, seq_length]
        # extended_attention_mask shape: [batch_size, 1, 1, seq_length]
        extended_attention_mask = extended_attention_mask.unsqueeze(1).unsqueeze(2)
        logging.debug("extended_attention_mask shape1: {}".format(extended_attention_mask.shape))

        if attention_mask is not None :
            ## 如果传进来的注意力mask不是None，那就直接用传进来的注意力mask 乘 原始mask
            # 注意 原始mask是extended_attention_mask，这个是用来把pad部分置为0，去掉pad部分影响
            # attention_mask shape: [batch_size, 1, seq_length, seq_length]
            extended_attention_mask = attention_mask * extended_attention_mask
            logging.debug("attention_mask shape: {}".format(attention_mask.shape))

        # extended_attention_mask shape: [batch_size, 1, seq_length, seq_length]
        logging.debug("extended_attention_mask shape2: {}".format(extended_attention_mask.shape))

        # 如果token_type_ids为None则默认为共一句话
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        # token_type_ids shape: [batch_size, seq_length]
        logging.debug("token_type_ids shape: {}".format(token_type_ids.shape))

        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        # 被mask的部分为 -10000.0，未mask的部分未0
        # extended_attention_mask会直接与raw_attention_score相加 然后过softmax操作
        # 则被mask的部分会极其小 相当于被忽略
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # extended_attention_mask shape: [batch_size, 1, seq_length, seq_length]
        logging.debug("extended_attention_mask shape3: {}".format(extended_attention_mask.shape))

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )
        # embedding_output shape: [batch_size, seq_length, config.hidden_size]
        logging.debug("embedding_output shape: {}".format(embedding_output.shape))

        # 返回config.num_hidden_layers层的输出
        encoder_layers, all_attention_matrices = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            output_attentions=output_attentions
        )
        logging.debug("encoder_layers size: {}".format(len(encoder_layers)))
        logging.debug("all_attention_matrices size: {}".format(len(all_attention_matrices)))

        # sequence_output shape: [batch_size, seq_length, config.hidden_size]
        sequence_output = encoder_layers[-1]
        logging.debug("sequence_output shape: {}".format(sequence_output.shape))

        # pooled_output shape: [batch_size, config.hidden_size]
        pooled_output = self.pooler(sequence_output)
        logging.debug("pooled_output shape: {}".format(pooled_output.shape))

        if output_attentions:
            return all_attention_matrices

        if not output_all_encoded_layers:
            # 如果不用输出所有encoder层
            encoder_layers = encoder_layers[-1]

        # 返回各层的输出 和最后层池化结果
        return encoder_layers, pooled_output


class BertForClassification(BertPreTrainedModel):
    """
    """
    def __init__(self, config):
        super(BertForClassification, self).__init__(config)
        self.bert = BertModel(self.config)
        self.final_fc = torch.nn.Linear(
                self.config.hidden_size,
                self.config.num_class)

    def compute_loss(self, predictions, labels):
        """
        计算loss
        predictions: (batch_size, 1)
        """
        predictions = predictions.view(-1, self.config.num_class)
        labels = labels.view(-1)
        loss = nn.CrossEntropyLoss(reduction="mean")
        return loss(predictions, labels)

    def forward(self, token_ids, token_type_ids=None, position_ids=None, labels=None, use_layer_num=-1):
        # 默认-1 取最后一层
        if use_layer_num != -1:
            if use_layer_num < 0 or use_layer_num > 7:
                # 越界
                raise Exception("层数选择错误，因为bert base模型共8层，"\
                    "所以参数只只允许0 - 7， 默认为-1，取最后一层")
        enc_layers, _ = self.bert(token_ids, token_type_ids=token_type_ids,
            position_ids=position_ids, output_all_encoded_layers=True)

        squence_out = enc_layers[use_layer_num]

        cls_token = squence_out[:, 0]# 取出cls向量 进行分类

        predictions = self.final_fc(cls_token)
        if labels is not None:
            ## 计算loss
            loss = self.compute_loss(predictions, labels)
            return loss, predictions
        else :
            return predictions


class BertForSeq2seq(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSeq2seq, self).__init__(config)
        self.bert = BertModel(self.config)
        self.decoder = BertLMPredictionHead(self.config, self.bert.embeddings.word_embeddings.weight)
        self.vocab_size = self.config.vocab_size

    def forward(self,
            input_ids,
            token_type_ids,
            position_ids=None,
            labels=None,
            output_all_encoded_layers=True,
            output_attentions=False,
            is_train=True,
            device="cpu"):
        if is_train:
            self.bert.train()
            self.decoder.train()
        else:
            self.bert.eval()
            self.decoder.eval()

        # input_tensor shape: [batch_size, seq_length]
        logging.debug("input_ids shape: {}".format(input_ids.shape))
        #logging.debug("input_tensor[0]: {}".format(input_tensor[0]))
        #logging.debug("input_tensor text: {}".format("/ ".join([self.ix2word[x] for x in input_tensor[0].cpu().numpy()])))

        # token_type_id 句子a及pad的部分都为0
        # token_type_id shape: [batch_size, seq_length]
        logging.debug("token_type_ids shape: {}".format(token_type_ids.shape))
        #logging.debug("token_type_ids[0]: {}".format(token_type_ids[0]))
        # position_ids shape: [batch_size, seq_length]
        if position_ids is not None:
            logging.debug("position_ids shape: {}".format(position_ids.shape))

        if labels is not None:
            logging.debug("labels shape: {}".format(labels.shape))

        ## 传入输入，位置编码，token type id ，还有句子a 和句子b的长度，注意都是传入一个batch数据
        ##  传入的几个值，在seq2seq 的batch iter 函数里面都可以返回
        input_shape = input_ids.shape
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        ## 构建特殊的mask
        ones = torch.ones((1, 1, seq_len, seq_len), dtype=torch.float32, device=input_ids.device)
        # a_mask shape: [1, 1, seq_length, seq_length]
        a_mask = ones.tril() # 下三角矩阵

        # 设text_a长度为len_a，text_b长度为len_b
        # 则token_type_id该行 = [0, 0, ..., 0, 1, 1, ...,1]
        # 其中0有len_a个，1有len_b个

        # s_ex12 shape: [batch_size, 1, 1, seq_length]
        # 1 - 行mask
        s_ex12 = token_type_ids.unsqueeze(1).unsqueeze(2).float()

        # 1 - 列mask
        # s_ex12 shape: [batch_size, 1, seq_length, 1]
        s_ex13 = token_type_ids.unsqueeze(1).unsqueeze(3).float()

        # (1.0 - s_ex12) * (1.0 - s_ex13) shape: [batch_size, 1, seq_length, seq_length]
        # 行maxk*列mask即attention时始终需要的元素
        # 在最后两维[seq_length, seq_length]的矩阵中
        # 左上方[len_a, len_a]大小的矩阵元素为1，其余元素均为0

        # s_ex13 * a_mask shape: [batch_size, 1, seq_length, seq_length]
        # 下三角矩阵*(1-列mask)主要是为了训练时不能得到后面的信息
        # 在最后两维[seq_length, seq_length]的矩阵中
        # 前len_a行全为0，后len_b行，前len_a全为1，右下角[len_b, len_b]的矩阵为下三角矩阵
        # a_mask shape: [batch_size, 1, seq_length, seq_length]
        a_mask = (1.0 - s_ex12) * (1.0 - s_ex13) + s_ex13 * a_mask 

        # 各层的输出shape：[batch_size, seq_length, config.hidden_size]
        enc_layers, _ = self.bert(
                input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                attention_mask=a_mask,
                output_all_encoded_layers=output_all_encoded_layers,
                output_attentions=output_attentions)
        # 取最后一层
        # sequence_output shape: [batch_size, seq_length, config.hidden_size]
        squence_out = enc_layers[-1]

        # predictions shape: [batch_size, seq_length, self.vocab_size]
        predictions = self.decoder(squence_out)

        if labels is not None:
            ## 计算loss
            ## 需要构建特殊的输出mask 才能计算正确的loss
            # 预测的值不用取最后sep符号的结果 因此是到-1

            # 去除最后一个预测的值 在构建训练数据的时候 label也会比输入的seq_length少1
            predictions = predictions[:, :-1].contiguous()
            # 计算loss时只考虑target_mask中为1的 这里去除第一个为0的 与predictions和labels的维度对齐
            # 这里取前面 是因为在构造label时 也是去掉了前一个 以达到原输入左移一位
            # 此时labels就是各位置的下一个位置需要输出的vocab的列表
            # 所以这里也应该从前面去除
            # 效果是 target_mask本来是从第一个SEP（不包含SEP）之后为1 表示第二句话开始
            # 现在前面去除一个
            # 则target_mask从第一个SEP（包含SEP）开始为1，表示从第一个SEP开始就要预测下一个vocab是什么
            target_mask = token_type_ids[:, 1:].contiguous()
            logging.debug("target_mask[0]: {}".format(target_mask[0]))

            loss = self.compute_loss(predictions, labels, target_mask)
            return predictions, loss
        else :
            return predictions

    def compute_loss(self, predictions, labels, target_mask):
        """
        target_mask : 句子a部分和pad部分全为0， 而句子b部分为1
        """
        # 打平
        # predictions shape: [batch_size*(seq_length-1), self.vocab_size]
        predictions = predictions.view(-1, self.vocab_size)
        logging.debug("predictions shape: {}".format(predictions.shape))

        # 打平
        # labels shape: [batch_size*(seq_length-1)]
        labels = labels.view(-1)
        logging.debug("labels shape: {}".format(labels.shape))

        # 只有text_b 即为1的需要预测
        target_mask = target_mask.view(-1).float()
        # 计算loss
        loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        # 通过mask 取消 pad 和句子a部分预测的影响
        return (loss(predictions, labels) * target_mask).sum() / target_mask.sum() 
