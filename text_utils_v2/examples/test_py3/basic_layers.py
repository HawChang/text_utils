#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   basic_layers.py
Author:   zhanghao55@baidu.com
Date  :   20/12/29 15:19:38
Desc  :   
"""

import sys
import collections
import logging
import numpy as np
import paddle.fluid as F
import paddle.fluid.dygraph as D
import paddle.fluid.layers as L
import six
import layer_utils as U

from functools import partial
from functools import reduce


class EmbeddingLayer(D.Layer):
    """
    Embedding Layer class
    """

    def __init__(self, vocab_size, emb_dim, is_sparse=True,
            dtype="float32", name="emb", padding_idx=None):
        """初始
        """
        super(EmbeddingLayer, self).__init__()
        self.emb_layer = D.Embedding(
            size=[vocab_size, emb_dim],
            dtype=dtype,
            is_sparse=is_sparse,
            padding_idx=padding_idx,
            param_attr=F.ParamAttr(
                name=name, initializer=F.initializer.Xavier()))

    def forward(self, inputs):
        """前向预测
        """
        return self.emb_layer(inputs)


class ConvPoolLayer(D.Layer):
    """卷积池化层
    """
    def __init__(self,
            num_channels,
            num_filters,
            filter_size,
            padding,
            use_cudnn=False,
            ):
        super(ConvPoolLayer, self).__init__()

        self._conv2d = D.Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            padding=padding,
            use_cudnn=use_cudnn,
            act='tanh')

    def forward(self, inputs):
        """前向预测
        """
        # inputs shape = [batch_size, num_channels, seq_len, emb_dim] [N, C, H, W]
        #print("inputs shape: {}".format(inputs.shape))

        # x shape = [batch_size, num_filters, height_after_conv, width_after_conv=1]
        x = self._conv2d(inputs)
        #print("conv3d shape: {}".format(x.shape))

        # x shape = [batch_size, num_filters, height_after_pool=1, width_after_pool=1]
        x = L.reduce_max(x, dim=2, keep_dim=True)
        #print("reduce sum shape: {}".format(x.shape))

        # x shape = [batch_size, num_filters]
        #x = L.squeeze(x, axes=[2, 3])
        x = L.reshape(x, shape=[x.shape[0], -1])
        return x


class TextCNNLayer(D.Layer):
    """textcnn层
    """
    def __init__(self,
            emb_dim,
            num_filters=10,
            num_channels=1,
            win_size_list=None,
            use_cudnn=True,
            ):
        super(TextCNNLayer, self).__init__()

        if win_size_list is None:
            win_size_list = [3]

        def gen_conv_pool(win_size):
            """生成指定窗口的卷积池化层
            """
            return ConvPoolLayer(
                    num_channels,
                    num_filters,
                    [win_size, emb_dim],
                    padding=[1, 0],
                    use_cudnn=use_cudnn,
                    )

        self.conv_pool_list = D.LayerList([gen_conv_pool(win_size) for win_size in win_size_list])


    def forward(self, input_emb):
        """前向预测
        """
        # emb shape = [batch_size, 1, seq_len, emb_dim]
        emb = L.unsqueeze(input_emb, axes=[1])
        #print("emb shape: {}".format(emb.shape))

        conv_pool_res_list = [conv_pool(emb) for conv_pool in self.conv_pool_list]

        conv_pool_res = L.concat(conv_pool_res_list, axis=-1)

        return conv_pool_res


class DynamicGRULayer(D.Layer):
    """动态GRU层
    """
    def __init__(self,
                 size,
                 param_attr=None,
                 bias_attr=None,
                 is_reverse=False,
                 gate_activation='sigmoid',
                 candidate_activation='tanh',
                 h_0=None,
                 origin_mode=False):
        super(DynamicGRULayer, self).__init__()
        self.gru_unit = D.GRUUnit(
            size * 3,
            param_attr=param_attr,
            bias_attr=bias_attr,
            activation=candidate_activation,
            gate_activation=gate_activation,
            origin_mode=origin_mode)
        self.size = size
        self.h_0 = h_0
        self.is_reverse = is_reverse

    def forward(self, inputs):
        """前向预测
        """
        #print("inputs shape: {}".format(inputs.shape))
        # 初始化h_0
        if self.h_0 is None:
            batch_size, _, _ = inputs.shape
            self.h_0 = D.to_variable(
                    np.zeros((batch_size, self.size), dtype="float32"))

        hidden = self.h_0
        res = []
        input_trans = L.transpose(inputs, perm=[1, 0, 2])
        #print("input trans shape: {}".format(input_trans.shape))
        for i in range(input_trans.shape[0]):
            if self.is_reverse:
                i = input_trans.shape[0] - 1 - i
            cur_input = input_trans[i]

            #print("cur input shape: {}".format(cur_input.shape))
            hidden, reset, gate = self.gru_unit(cur_input, hidden)
            res.append(L.unsqueeze(input=hidden, axes=[1]))
        if self.is_reverse:
            res = res[::-1]
        res = L.concat(res, axis=1)
        #print("res shape: {}".format(res.shape))

        return res


class RNNUnit(D.Layer):
    def get_initial_states(self,
                           batch_ref,
                           shape=None,
                           dtype=None,
                           init_value=0,
                           batch_dim_idx=0):
        """
        Generate initialized states according to provided shape, data type and
        value.

        Parameters:
            batch_ref: A (possibly nested structure of) tensor variable[s].
                The first dimension of the tensor will be used as batch size to
                initialize states.
            shape: A (possiblely nested structure of) shape[s], where a shape is
                represented as a list/tuple of integer). -1(for batch size) will
                beautomatically inserted if shape is not started with it. If None,
                property `state_shape` will be used. The default value is None.
            dtype: A (possiblely nested structure of) data type[s]. The structure
                must be same as that of `shape`, except when all tensors' in states
                has the same data type, a single data type can be used. If None and
                property `cell.state_shape` is not available, float32 will be used
                as the data type. The default value is None.
            init_value: A float value used to initialize states.

        Returns:
            Variable: tensor variable[s] packed in the same structure provided \
                by shape, representing the initialized states.
        """
        # TODO: use inputs and batch_size
        batch_ref = U.flatten(batch_ref)[0]

        def _is_shape_sequence(seq):
            if sys.version_info < (3, ):
                integer_types = (
                    int,
                    long, )
            else:
                integer_types = (int, )
            """For shape, list/tuple of integer is the finest-grained objection"""
            # 如果是列表或元组 其元素类型应该在integer_types之类
            if (isinstance(seq, list) or isinstance(seq, tuple)):
                # reduce(func, sequence, initial)
                if reduce(lambda flag, x: isinstance(x, integer_types) and flag,
                          seq, True):
                    return False
            # 可以是字典
            # TODO: Add check for the illegal
            if isinstance(seq, dict):
                return True

            # 可以是collections.Sequence 且不是字符串类型即可
            return (isinstance(seq, collections.Sequence) and
                    not isinstance(seq, six.string_types))

        class Shape(object):
            def __init__(self, shape):
                self.shape = shape if shape[0] == -1 else ([-1] + list(shape))

        # nested structure of shapes
        states_shapes = self.state_shape if shape is None else shape
        is_sequence_ori = U.is_sequence
        U.is_sequence = _is_shape_sequence
        states_shapes = U.map_structure(lambda shape: Shape(shape), states_shapes)
        U.is_sequence = is_sequence_ori

        # nested structure of dtypes
        try:
            states_dtypes = self.state_dtype if dtype is None else dtype
        except NotImplementedError:  # use fp32 as default
            states_dtypes = "float32"
        if len(U.flatten(states_dtypes)) == 1:
            dtype = U.flatten(states_dtypes)[0]
            states_dtypes = U.map_structure(lambda shape: dtype, states_shapes)

        init_states = U.map_structure(
            lambda shape, dtype: L.fill_constant_batch_size_like(
                input=batch_ref,
                shape=shape.shape,
                dtype=dtype,
                value=init_value,
                input_dim_idx=batch_dim_idx), states_shapes, states_dtypes)
        return init_states

    @property
    def state_shape(self):
        """
        Abstract method (property).
        Used to initialize states.
        A (possiblely nested structure of) shape[s], where a shape is represented
        as a list/tuple of integers (-1 for batch size would be automatically
        inserted into a shape if shape is not started with it).
        Not necessary to be implemented if states are not initialized by
        `get_initial_states` or the `shape` argument is provided when using
        `get_initial_states`.
        """
        raise NotImplementedError(
            "Please add implementaion for `state_shape` in the used cell.")

    @property
    def state_dtype(self):
        """
        Abstract method (property).
        Used to initialize states.
        A (possiblely nested structure of) data types[s]. The structure must be
        same as that of `shape`, except when all tensors' in states has the same
        data type, a signle data type can be used.
        Not necessary to be implemented if states are not initialized
        by `get_initial_states` or the `dtype` argument is provided when using
        `get_initial_states`.
        """
        raise NotImplementedError(
            "Please add implementaion for `state_dtype` in the used cell.")


class BasicLSTMUnit(RNNUnit):
    """
    ****
    BasicLSTMUnit class, Using basic operator to build LSTM
    The algorithm can be described as the code below.
        .. math::
           i_t &= \sigma(W_{ix}x_{t} + W_{ih}h_{t-1} + b_i)
           f_t &= \sigma(W_{fx}x_{t} + W_{fh}h_{t-1} + b_f + forget_bias )
           o_t &= \sigma(W_{ox}x_{t} + W_{oh}h_{t-1} + b_o)
           \\tilde{c_t} &= tanh(W_{cx}x_t + W_{ch}h_{t-1} + b_c)
           c_t &= f_t \odot c_{t-1} + i_t \odot \\tilde{c_t}
           h_t &= o_t \odot tanh(c_t)
        - $W$ terms denote weight matrices (e.g. $W_{ix}$ is the matrix
          of weights from the input gate to the input)
        - The b terms denote bias vectors ($bx_i$ and $bh_i$ are the input gate bias vector).
        - sigmoid is the logistic sigmoid function.
        - $i, f, o$ and $c$ are the input gate, forget gate, output gate,
          and cell activation vectors, respectively, all of which have the same size as
          the cell output activation vector $h$.
        - The :math:`\odot` is the element-wise product of the vectors.
        - :math:`tanh` is the activation functions.
        - :math:`\\tilde{c_t}` is also called candidate hidden state,
          which is computed based on the current input and the previous hidden state.
    Args:
        hidden_size (integer): The hidden size used in the Unit.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
            weight matrix. Note:
            If it is set to None or one attribute of ParamAttr, lstm_unit will
            create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|None): The parameter attribute for the bias
            of LSTM unit.
            If it is set to None or one attribute of ParamAttr, lstm_unit will
            create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized as zero. Default: None.
        gate_activation (function|None): The activation function for gates (actGate).
                                  Default: 'fluid.layers.sigmoid'
        activation (function|None): The activation function for cells (actNode).
                             Default: 'fluid.layers.tanh'
        forget_bias(float|1.0): forget bias used when computing forget gate
        dtype(string): data type used in this unit
    """

    def __init__(self,
                 hidden_size,
                 input_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation=None,
                 activation=None,
                 forget_bias=1.0,
                 dtype='float32'):
        super(BasicLSTMUnit, self).__init__(dtype)

        #logging.info("input size: {}".format(input_size))
        #logging.info("hidden size: {}".format(hidden_size))
        self._hidden_size = hidden_size
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._gate_activation = gate_activation or L.sigmoid
        self._activation = activation or L.tanh
        self._forget_bias = L.fill_constant(
            [1], dtype=dtype, value=forget_bias)
        self._forget_bias.stop_gradient = False
        self._dtype = dtype
        self._input_size = input_size

        # weight shape = [input_size+hidden_size, 4*hidden_size]
        self._weight = self.create_parameter(
            attr=self._param_attr,
            shape=[
                self._input_size + self._hidden_size, 4 * self._hidden_size
            ],
            dtype=self._dtype)

        # weight shape = [4*hidden_size,]
        self._bias = self.create_parameter(
            attr=self._bias_attr,
            shape=[4 * self._hidden_size],
            dtype=self._dtype,
            is_bias=True)

    def forward(self, input, state):
        #logging.info("input shape: {}".format(input.shape))
        pre_hidden, pre_cell = state
        #logging.info("pre hidden shape: {}".format(pre_hidden.shape))
        #logging.info("pre cell shape: {}".format(pre_cell.shape))
        # i,f,c,o 四个值均有Wx+Wh+b 即W(x+h)+b
        # 因此：
        # 实际相乘为[x, b]·W+b
        # x,b 横向相连, shape为[batch_size, input_size+hidden_size]
        # W的shape为[input_size+hidden_size, 4*hidden_size]
        # b的shape为[4*hidden_size,]

        # 横向连接
        # shape: [batch_size, input_size+hidden_size]
        concat_input_hidden = L.concat([input, pre_hidden], axis=1)
        #logging.info("x concat h shape: {}".format(concat_input_hidden.shape))

        # 计算Wx+Wh+b
        # shape: [batch_size, 4*hidden_size]
        gate_input = L.matmul(x=concat_input_hidden, y=self._weight)
        #logging.info("[x, b]·W shape: {}".format(gate_input.shape))

        # shape: [batch_size, 4*hidden_size]
        gate_input = L.elementwise_add(gate_input, self._bias)
        #logging.info("[x, b]·W+b shape: {}".format(gate_input.shape))

        # i,f,c,o四值按最后一维分开 因此每个的最后一维都是hidden_size
        i, f, c, o = L.split(gate_input, num_or_sections=4, dim=-1)

        # new_c = pre_c·sigmoid(f+forget_bias) + sigmoid(i)·tanh(c)
        # shape: [batch_size, hidden_size]
        new_cell = L.elementwise_add(
            L.elementwise_mul(
                pre_cell,
                L.sigmoid(L.elementwise_add(f, self._forget_bias))),
            L.elementwise_mul(L.sigmoid(i), L.tanh(c))
            )
        #logging.info("new_cell shape: {}".format(new_cell.shape))

        # new_h = tanh(new_c)*sigmoid(o)
        # shape: [batch_size, hidden_size]
        new_hidden = L.tanh(new_cell) * L.sigmoid(o)
        #logging.info("new_hidden shape: {}".format(new_hidden.shape))

        return new_hidden, [new_hidden, new_cell]

    @property
    def state_shape(self):
        # state形状
        return [[self._hidden_size], [self._hidden_size]]


class RNN(D.Layer):
    def __init__(self, cell, is_reverse=False, time_major=False, **kwargs):
        super(RNN, self).__init__()
        self.cell = cell
        if not hasattr(self.cell, "call"):
            self.cell.call = self.cell.forward
        self.is_reverse = is_reverse
        self.time_major = time_major
        self.batch_index, self.time_step_index = (1, 0) \
                if time_major else (0, 1)

    def forward(self,
                inputs,
                initial_states=None,
                sequence_length=None,
                **kwargs):
        if F.in_dygraph_mode():

            class OutputArray(object):
                def __init__(self, x):
                    self.array = [x]

                def append(self, x):
                    self.array.append(x)

            def _maybe_copy(state, new_state, step_mask):
                # TODO: use where_op
                new_state = L.elementwise_mul(new_state, step_mask, axis=0) - \
                        L.elementwise_mul(state, (step_mask - 1), axis=0)
                return new_state

            #logging.info("inputs shape: {}".format(inputs.shape))
            flat_inputs = U.flatten(inputs)
            #logging.info("flat inputs len: {}".format(len(flat_inputs)))
            #logging.info("flat inputs[0] shape: {}".format(flat_inputs[0].shape))

            batch_size, time_steps = (
                flat_inputs[0].shape[self.batch_index],
                flat_inputs[0].shape[self.time_step_index])
            #logging.info("batch_size: {}".format(batch_size))
            #logging.info("time_steps: {}".format(time_steps))

            if initial_states is None:
                initial_states = self.cell.get_initial_states(
                    batch_ref=inputs, batch_dim_idx=self.batch_index)

            if not self.time_major:
                # 如果第一维不是时间步 则第一维和第二维交换
                # 第一维为时间步
                inputs = U.map_structure(
                    lambda x: L.transpose(x, [1, 0] + list(
                        range(2, len(x.shape)))), inputs)

            if sequence_length is not None:
                mask = L.sequence_mask(
                    sequence_length,
                    maxlen=time_steps,
                    dtype=U.flatten(initial_states)[0].dtype)
                # 同样 第一维为时间步
                mask = L.transpose(mask, [1, 0])

            if self.is_reverse:
                # 如果反向
                # 则第一维反向
                inputs = U.map_structure(lambda x: L.reverse(x, axis=[0]), inputs)
                mask = L.reverse(mask, axis=[0]) if sequence_length is not None else None

            states = initial_states
            outputs = []
            # 遍历时间步
            for i in range(time_steps):
                # 取该时间步的输入
                step_inputs = U.map_structure(lambda x: x[i], inputs)
                # 输入当前输入和状态
                # 得到输出和新状态
                step_outputs, new_states = self.cell(step_inputs, states, **kwargs)
                if sequence_length is not None:
                    # 如果有mask 则被mask的地方 用原state的数
                    # _maybe_copy: 未mask的部分用new_states, 被mask的部分用states
                    new_states = U.map_structure(
                        partial(_maybe_copy, step_mask=mask[i]),
                        states,
                        new_states)
                states = new_states
                #logging.info("step_output shape: {}".format(step_outputs.shape))

                if i == 0:
                    # 初始时，各输出
                    outputs = U.map_structure(lambda x: OutputArray(x), step_outputs)
                else:
                    # 各输出加入对应list中
                    U.map_structure(lambda x, x_array: x_array.append(x), step_outputs, outputs)

            # 最后按时间步的维度堆叠
            final_outputs = U.map_structure(
                lambda x: L.stack(x.array, axis=self.time_step_index),
                outputs)
            #logging.info("final_outputs shape: {}".format(final_outputs.shape))

            if self.is_reverse:
                # 如果是反向 则最后结果也反向一下
                final_outputs = U.map_structure(
                    lambda x: L.reverse(x, axis=self.time_step_index),
                    final_outputs)

            final_states = new_states

        else:
            final_outputs, final_states = L.rnn(
                self.cell,
                inputs,
                initial_states=initial_states,
                sequence_length=sequence_length,
                time_major=self.time_major,
                is_reverse=self.is_reverse,
                **kwargs)

        return final_outputs, final_states


class DynamicLSTMLayer(D.Layer):
    """
    Dynamic LSTM Layer class
    """

    def __init__(self, input_size, hidden_size, is_reverse=False, name="dyn_lstm", time_major=False):
        """初始化
        """
        super(DynamicLSTMLayer, self).__init__()
        lstm_cell = BasicLSTMUnit(input_size=input_size, hidden_size=hidden_size)
        self.lstm = RNN(cell=lstm_cell, time_major=time_major, is_reverse=is_reverse)

    def forward(self, inputs):
        """前向预测
        """
        return self.lstm(inputs)


class EncoderCell(RNNUnit):
    def __init__(self, num_layers, input_size, hidden_size, dropout_prob=0.):
        super(EncoderCell, self).__init__()
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        self.lstm_cells = list()
        for i in range(self.num_layers):
            self.lstm_cells.append(
                self.add_sublayer("layer_%d" % i,
                                  BasicLSTMUnit(input_size if i == 0 else
                                                hidden_size, hidden_size)))

    def forward(self, step_input, states):
        new_states = []
        for i in range(self.num_layers):
            out, new_state = self.lstm_cells[i](step_input, states[i])
            step_input = L.dropout(
                out, self.dropout_prob) if self.dropout_prob > 0 else out
            new_states.append(new_state)
        return step_input, new_states

    @property
    def state_shape(self):
        return [cell.state_shape for cell in self.lstm_cells]
