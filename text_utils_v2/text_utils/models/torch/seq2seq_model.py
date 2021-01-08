import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from text_utils.tokenizers.bert_tokenizer import Tokenizer
import time
from text_utils.models.torch.config import yayun_list
from text_utils.models.torch.bert_model import BertConfig, BertModel, BertLMPredictionHead
import os 


class Seq2SeqModel(nn.Module):
    """
    """
    def __init__(self, config):
        super(Seq2SeqNet, self).__init__()

        assert "tokenizer" in config, "tokenizer is required to init Seq2SeqNet"
        self.tokenizer = config["tokenizer"]
        self.word2ix = self.tokenizer._token_dict
        self.vocab_size = len(self.word2ix)
        #self.word2ix = word2ix
        #self.vocab_size = len(word2ix)
        #self.tokenizer = Tokenizer(word2ix)

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
        logging.debug("predictions shape: {}".format(labels.shape))

        # 只有text_b 即为1的需要预测
        target_mask = target_mask.view(-1).float()
        # 计算loss
        loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")

        return (loss(predictions, labels) * target_mask).sum() / target_mask.sum() ## 通过mask 取消 pad 和句子a部分预测的影响

    #def forward(self, input_tensor, token_type_id, position_enc=None, labels=None, device="cpu"):
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def generate(self, text, out_max_length=40, beam_size=1, device="cpu", is_poem=False, max_length=256):
        # 对 一个 句子生成相应的结果
        ## 通过输出最大长度得到输入的最大长度，这里问题不大，如果超过最大长度会进行截断
        self.out_max_length = out_max_length
        input_max_length = max_length - out_max_length

        # print(text)
        # token_type_id 全为0
        token_ids, token_type_ids = self.tokenizer.encode(text, max_length=input_max_length)
        token_ids = torch.tensor(token_ids, device=device).view(1, -1)
        token_type_ids = torch.tensor(token_type_ids, device=device).view(1, -1)
        if is_poem:## 古诗的beam-search稍有不同
            logging.debug("poem beam_search")
            out_puts_ids = self.beam_search_poem(text, token_ids, token_type_ids, self.word2ix, beam_size=beam_size, device=device)
        else:
            logging.debug("common beam_search")
            out_puts_ids = self.beam_search(token_ids, token_type_ids, self.word2ix, beam_size=beam_size, device=device)

        # 解码 得到相应输出
        # if err is False:
        #     return self.tokenizer.decode(out_puts_ids)

        return self.tokenizer.decode(out_puts_ids.cpu().numpy())

    # def poem_beam_search(self, token_ids, token_type_ids, word2ix, beam_size=1, device="cpu"):
    #     """
    #     专门针对写诗的beam-search
    #     """
    #     ix2word = {v: k for k, v in word2ix.items()}
    #     sep_id = word2ix["[SEP]"]
    #     douhao_id = word2ix["，"]# 逗号
    #     juhao_id = word2ix["。"]# 句号
    #     # 用来保存输出序列
    #     output_ids = [[]]
    #     # word_list = {} # 保证不重复生成
    #     repeat_list = [[], [], [], [], []]
    #     last_chars = []
    #     yayun_save = -1
    #     # 用来保存累计得分
    #     output_scores = torch.zeros(token_ids.shape[0], device=device)
    #     flag = 0 # 判断第一次遇到逗号
    #     for step in range(self.out_max_length):
    #         scores = self.forward(token_ids, token_type_ids, device=device)
    #         if step == 0:
    #             # 重复beam-size次 输入ids
    #             token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
    #             token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
    #         ## 计算log 分值 (beam_size, vocab_size)
    #         logit_score = torch.log_softmax(scores, dim=-1)[:, -1]
    #         logit_score = output_scores.view(-1, 1) + logit_score # 累计得分
    #         ## 取topk的时候我们是展平了然后再去调用topk函数
    #         # 展平
    #         logit_score = logit_score.view(-1)
    #         hype_score, hype_pos = torch.topk(logit_score, beam_size)
    #         indice1 = hype_pos // scores.shape[-1] # 行索引
    #         indice2 = hype_pos % scores.shape[-1] # 列索引

    #         # 下面需要更新一下输出了
    #         new_hype_scores = []
    #         new_hype_ids = []
    #         new_repeat_list = []
    #         next_chars = [] # 用来保存新预测出来的一个字符，继续接到输入序列后面，再去预测新字符
    #         index = 0
    #         for i_1, i_2, score in zip(indice1, indice2, hype_score):
    #             i_1 = i_1.item()
    #             i_2 = i_2.item()
    #             score = score.item()
    #             if i_2 != douhao_id and i_2 != juhao_id:
    #                 if i_2 in repeat_list[i_1]:
    #                 # 说明出现重复了
    #                 # 扣分
    #                     score -= 1
    #                     hype_score[i_1] -= 1
    #                 else :
    #                     repeat_list[i_1].append(i_2)

    #                 # if i_2 not in word_list.keys():
    #                 #     word_list[i_2] = 1
    #                 # else :
    #                 #     # 加惩罚
    #                 #     word_list[i_2] += 1
    #                 #     score -= 1 * word_list[i_2]
    #                 #     hype_score[index] -= 1 * word_list[i_2]
    #             if flag == 0 and i_2 == douhao_id and len(last_chars) != 0:
                    
    #                 flag += 1
    #                 word = ix2word[last_chars[index]]# 找到上一个字符 记住其押韵情况
    #                 for i, each_yayun in enumerate(yayun_list):
    #                     if word in each_yayun:
    #                         yayun_save = i
    #                         break
    #             if i_2 == juhao_id and len(last_chars) != 0:

    #                 word = ix2word[last_chars[i_1]]
    #                 # 找押韵 给奖励
    #                 if word in yayun_list[yayun_save]:
    #                     score += 2
    #                     hype_score[i_1] += 2
    #                 else:
    #                     score -= 2
    #                     hype_score[i_1] -= 2
    #             hype_id = output_ids[i_1] + [i_2] # 保存所有输出的序列，而不仅仅是新预测的单个字符

    #             if i_2 == sep_id:
    #                 # 说明解码到最后了
    #                 if score == torch.max(hype_score).item():
    #                     return hype_id[: -1], False
    #                 else:
    #                     # 完成一个解码了，但这个解码得分并不是最高，因此的话需要跳过这个序列
    #                     beam_size -= 1
    #             else :
    #                 new_hype_ids.append(hype_id)
    #                 new_hype_scores.append(score)
    #                 next_chars.append(i_2) # 收集一下，需要连接到当前的输入序列之后
    #                 new_repeat_list.append(repeat_list[i_1])
    #             index += 1

    #         output_ids = new_hype_ids
    #         repeat_list = new_repeat_list ## 重复会扣分
    #         last_chars = next_chars.copy() # 记录一下上一个字符
    #         output_scores = torch.tensor(new_hype_scores, dtype=torch.float32, device=device)
    #         # 现在需要重新构造输入数据了，用上一次输入连接上这次新输出的字符，再输入bert中预测新字符
    #         token_ids = token_ids[:len(output_ids)].contiguous() # 截取，因为要过滤掉已经完成预测的序列
    #         token_type_ids = token_type_ids[: len(output_ids)].contiguous()

    #         next_chars = torch.tensor(next_chars, dtype=torch.long, device=device).view(-1, 1)
    #         next_token_type_ids = torch.ones_like(next_chars, device=device)
    #         # 连接
    #         token_ids = torch.cat((token_ids, next_chars), dim=1)
    #         token_type_ids = torch.cat((token_type_ids, next_token_type_ids), dim=1)
    #         if beam_size < 1:
    #             break

    #     # 如果达到最大长度的话 直接把得分最高的输出序列返回把
    #     err = False
    #     try: 
    #         return output_ids[output_scores.argmax().item()], err
    #     except:
    #         err = True
    #         return "本次解码出现错误", err

    def beam_search(self, token_ids, token_type_ids, word2ix, beam_size=1, device="cpu"):
        """
        beam-search操作
        """
        # 一次只输入一个
        # batch_size = 1

        # token_ids shape: [batch_size, seq_length]
        logging.debug("token_ids: {}".format(token_ids))
        logging.debug("token_ids shape: {}".format(token_ids.shape))

        # token_type_ids shape: [batch_size, seq_length]
        logging.debug("token_type_ids : {}".format(token_type_ids))
        logging.debug("token_type_ids  shape: {}".format(token_type_ids.shape))

        sep_id = word2ix["[SEP]"]

        # 用来保存输出序列
        output_ids = torch.empty(1, 0, device=device, dtype=torch.long)
        logging.debug("output_ids: {}".format(output_ids))
        logging.debug("output_ids shape: {}".format(output_ids.shape))
        # 用来保存累计得分

        with torch.no_grad():
            # 初始化各得分
            # output_scores shape: [batch_size]
            output_scores = torch.zeros(token_ids.shape[0], device=device)
            # 重复生成 直到达到最大长度
            for step in range(self.out_max_length):
                logging.debug("beam size: {}".format(beam_size))
                if step == 0:
                    # score shape: [batch_size, seq_length, self.vocab_size]
                    scores = self.forward(token_ids, token_type_ids, device=device, is_train=False)
                    logging.debug("scores shape: {}".format(scores.shape))

                    # 重复beam-size次 输入ids
                    # token_ids shape: [beam_size, batch_size*seq_length]
                    token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                    logging.debug("token_ids shape: {}".format(token_ids.shape))

                    # token_type_ids shape: [beam_size, batch_size*seq_length]
                    token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
                    logging.debug("token_type_ids shape: {}".format(token_type_ids.shape))
                else:
                    # TODO score shape: [beam_size, cur_seq_length, self.vocab_size]
                    # cur_seq_length是逐渐变化的
                    scores = self.forward(new_input_ids, new_token_type_ids, device=device, is_train=False)
                    logging.debug("scores shape: {}".format(scores.shape))

                # 只取最后一个输出在vocab上的score
                # logit_score shape: [batch_size, self.vocab_size]
                logit_score = torch.log_softmax(scores[:, -1], dim=-1)
                logging.debug("logit_score shape: {}".format(logit_score.shape))

                # logit_score shape: [batch_size, self.vocab_size]
                logit_score = output_scores.view(-1, 1) + logit_score # 累计得分
                logging.debug("logit_score shape: {}".format(logit_score.shape))

                ## 取topk的时候我们是展平了然后再去调用topk函数
                # 展平
                # 这是beam_size种结果各vocab的结果打平
                logit_score = logit_score.view(-1)
                # 找到topk的值和位置
                hype_score, hype_pos = torch.topk(logit_score, beam_size)

                # 根据打平后的位置 找到其打平前的行列位置
                # 行位置其实是beam_size中 的第几个beam 行位置可能有重复
                # 列位置其实是当前beam下的vocab_id vocab_id也可能有重复
                indice1 = (hype_pos // scores.shape[-1]) # 行索引
                logging.debug("indice1: {}".format(indice1))
                indice2 = (hype_pos % scores.shape[-1]).long().reshape(-1, 1) # 列索引
                logging.debug("indice2: {}".format(indice2))

                # 更新得分
                # output_scores shape: [beam_size]
                output_scores = hype_score
                logging.debug("output_scores: {}".format(output_scores))

                # 更新output_ids
                # 通过indice1选是哪个beam
                # 通过indice2选当前beam加哪个vocab
                # output_ids shape: [beam_size, cur_seq_length]
                output_ids = torch.cat([output_ids[indice1], indice2], dim=1).long()
                logging.debug("output_ids: {}".format(output_ids))

                # new_input_ids shape: [beam_size, cur_seq_length]
                # token_ids是固定原输入
                # output_ids是当前beam_search留下的beam_size个候选路径
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                logging.debug("new_input_ids shape: {}".format(new_input_ids.shape))

                # new_input_ids shape: [beam_size, cur_seq_length]
                # output_ids的type全为1
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids)], dim=1)
                logging.debug("new_token_type_ids shape: {}".format(new_token_type_ids.shape))

                # 记录当前output_ids中有sep_id的情况
                end_counts = (output_ids == sep_id).sum(1)  # 统计出现的end标记
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    # 如果当前分数最优的已结束 则选当前该beam
                    # 说明出现终止了～
                    return output_ids[best_one]
                else :
                    # 否则 去除出现sep_id的序列
                    flag = (end_counts < 1)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        token_ids = token_ids[flag]
                        token_type_ids = token_type_ids[flag]
                        new_input_ids = new_input_ids[flag]
                        new_token_type_ids = new_token_type_ids[flag]
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        end_counts = end_counts[flag]  # 扔掉已完成end计数
                        beam_size = flag.sum()  # topk相应变化
                        logging.debug("beam size change")

            return output_ids[output_scores.argmax()]

    def beam_search_poem(self, text, token_ids, token_type_ids, word2ix, beam_size=1, device="cpu"):
        """
        beam-search操作
        """
        yayun_pos = []
        title = text.split("##")[0]
        if "五言律诗" in text:
            yayun_pos = [10, 22, 34, 46]
        elif "五言绝句" in text:
            yayun_pos = [10, 22]
        elif "七言律诗" in text:
            yayun_pos = [14, 30, 46, 62]
        elif "七言绝句" in text:
            yayun_pos = [14, 30]
        sep_id = word2ix["[SEP]"]
        douhao_id = word2ix["，"]# 逗号
        ix2word = {v: k for k, v in word2ix.items()}
        juhao_id = word2ix["。"]# 句号
        repeat_word = [[] for i in range(beam_size)]
        # 用来保存输出序列
        output_ids = torch.empty(1, 0, device=device, dtype=torch.long)
        last_chars = None #torch.empty(1, 0, device=device, dtype=torch.long)
        yayun_chars = (-1) * torch.ones(beam_size, dtype=torch.long)
        start = 0
        with torch.no_grad(): 
            output_scores = torch.zeros(token_ids.shape[0], device=device)
            for step in range(self.out_max_length):
                #logging.debug("step #{}".format(step))
                #logging.debug("last_chars: {}".format(last_chars))
                if step == 0:
                    scores = self.forward(token_ids, token_type_ids, device=device)
                    # 重复beam-size次 输入ids
                    token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                    token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
                else:
                    scores = self.forward(new_input_ids, new_token_type_ids, device=device)

                logit_score = torch.log_softmax(scores[:, -1], dim=-1)

                if last_chars is not None:
                    for i, char in enumerate(last_chars):

                        for word in repeat_word[i]:
                            logit_score[i, word] -= 5
                        for word in title:
                            ix = word2ix.get(word, -1)
                            if ix != -1:
                                logit_score[i, ix] += 2

                if step in yayun_pos:
                    # print("step is " + str(step))
                    # print("yayun_chars is " + str(yayun_chars))
                    if last_chars is not None:
                        for i, char in enumerate(last_chars):
                            if yayun_chars[i].item() != -1:
                                yayuns = yayun_list[yayun_chars[i].item()]
                                for char in yayuns:
                                    ix = word2ix.get(char, -1)
                                    if ix != -1:
                                        # print("char is " + str(char))
                                        logit_score[i, ix] += 10


                logit_score = output_scores.view(-1, 1) + logit_score # 累计得分
                ## 取topk的时候我们是展平了然后再去调用topk函数
                # 展平
                logit_score = logit_score.view(-1)
                hype_score, hype_pos = torch.topk(logit_score, beam_size)
                indice1 = (hype_pos // scores.shape[-1]) # 行索引
                indice2 = (hype_pos % scores.shape[-1]).long().reshape(-1, 1) # 列索引

                for index, each_out in zip(indice1, indice2):
                    index = index.item()
                    each_out = each_out.item()
                    #logging.debug("cur_out: {}".format(ix2word[each_out]))

                    if each_out in repeat_word[index]:
                        pass 
                        # repeat_word[index].append(each_out)
                        # hype_score[index] -= 2 * repeat_word[index].count(each_out)
                    else :
                        repeat_word[index].append(each_out)

                    if start < beam_size and each_out == douhao_id and last_chars is not None:
                        start += 1
                        #logging.debug("last_chars[{}] = {}".format(index, last_chars[index]))
                        word = ix2word[last_chars[index].item()]# 找到上一个字符 记住其押韵情况
                        for i, each_yayun in enumerate(yayun_list):
                            if word in each_yayun:
                                yayun_chars[index] = i
                                break

                    # if each_out == juhao_id and len(last_chars) != 0:  
                    #     word = ix2word[last_chars[index].item()]
                    #     if yayun_chars[index].item() != -1 and word in yayun_list[yayun_chars[index].item()]:
                    #         hype_score[index] += 10
                    #     else:
                    #         hype_score[index] -= 5

                # 更新得分
                output_scores = hype_score

                last_chars = indice2

                output_ids = torch.cat([output_ids[indice1], indice2], dim=1).long()
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids)], dim=1)

                end_counts = (output_ids == sep_id).sum(1)  # 统计出现的end标记
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    # 说明出现终止了～
                    # print(repeat_word)
                    # print(yayun_chars)
                    return output_ids[best_one]
                else :
                    # 保留未完成部分
                    flag = (end_counts < 1)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        token_ids = token_ids[flag]
                        token_type_ids = token_type_ids[flag]
                        last_chars = last_chars[flag]
                        yayun_chars = yayun_chars[flag]
                        new_input_ids = new_input_ids[flag]
                        new_token_type_ids = new_token_type_ids[flag]
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        end_counts = end_counts[flag]  # 扔掉已完成end计数
                        beam_size = flag.sum()  # topk相应变化
                        flag = flag.long()

                        new_repeat_word = []
                        for index, i in enumerate(flag):
                            if i.item() == 1:
                                new_repeat_word.append(repeat_word[index])

                        repeat_word = new_repeat_word


            # print(repeat_word)
            # print(yayun_chars)
            return output_ids[output_scores.argmax()]

    def beam_search_poem_v2(self, text, token_ids, token_type_ids, word2ix, beam_size=1, device="cpu"):
        """
        beam-search操作
        """
        yayun_pos = []
        if "五言律诗" in text:
            yayun_pos = [10, 22, 34, 46]
        elif "五言绝句" in text:
            yayun_pos = [10, 22]
        elif "七言律诗" in text:
            yayun_pos = [14, 30, 46, 62]
        elif "七言绝句" in text:
            yayun_pos = [14, 30]
        sep_id = word2ix["[SEP]"]
        douhao_id = word2ix["，"]# 逗号
        ix2word = {v: k for k, v in word2ix.items()}
        juhao_id = word2ix["。"]# 句号
        repeat_word = []
        # 用来保存输出序列
        output_ids = torch.empty(1, 0, device=device, dtype=torch.long)
        last_chars = torch.empty(1, 0, device=device, dtype=torch.long)
        yayun_chars = (-1) * torch.ones(beam_size, dtype=torch.long)
        start = 0
        with torch.no_grad(): 
            output_scores = torch.zeros(token_ids.shape[0], device=device)
            for step in range(self.out_max_length):
                if step == 0:
                    scores = self.forward(token_ids, token_type_ids, device=device)
                    # 重复beam-size次 输入ids
                    token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                    token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
                else:
                    scores = self.forward(new_input_ids, new_token_type_ids, device=device)

                logit_score = torch.log_softmax(scores[:, -1], dim=-1)
                # if len(last_chars) != 0:
                #     logit_score[last_chars] -= 5
                for i, char in enumerate(last_chars):
                    logit_score[i, char] -= 2
                    for word in repeat_word:
                        logit_score[i, word] -= 1
                if step in yayun_pos:
                    # print("step is " + str(step))
                    # print("yayun_chars is " + str(yayun_chars))
                    for i, char in enumerate(last_chars):
                        if yayun_chars[i].item() != -1:
                            yayuns = yayun_list[yayun_chars[i].item()]
                            for char in yayuns:
                                ix = word2ix.get(char, -1)
                                if ix != -1:
                                    # print("char is " + str(char))
                                    logit_score[i, ix] += 3
                logit_score = output_scores.view(-1, 1) + logit_score # 累计得分
                ## 取topk的时候我们是展平了然后再去调用topk函数
                # 展平
                logit_score = logit_score.view(-1)
                hype_score, hype_pos = torch.topk(logit_score, beam_size)
                indice1 = (hype_pos // scores.shape[-1]) # 行索引
                indice2 = (hype_pos % scores.shape[-1]).long().reshape(-1, 1) # 列索引

                for index, each_out in zip(indice1, indice2):
                    index = index.item()
                    each_out = each_out.item()

                    if each_out in repeat_word:
                        pass 
                        # repeat_word[index].append(each_out)
                        # hype_score[index] -= 2 * repeat_word[index].count(each_out)
                    else :
                        repeat_word.append(each_out)

                    if start < beam_size and each_out == douhao_id and len(last_chars) != 0:
                        start += 1
                        word = ix2word[last_chars[index].item()]# 找到上一个字符 记住其押韵情况
                        for i, each_yayun in enumerate(yayun_list):
                            if word in each_yayun:
                                yayun_chars[index] = i
                                break

                    # if each_out == juhao_id and len(last_chars) != 0:  
                    #     word = ix2word[last_chars[index].item()]
                    #     if yayun_chars[index].item() != -1 and word in yayun_list[yayun_chars[index].item()]:
                    #         hype_score[index] += 10
                    #     else:
                    #         hype_score[index] -= 5

                # 更新得分
                output_scores = hype_score

                last_chars = indice2

                output_ids = torch.cat([output_ids[indice1], indice2], dim=1).long()
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids)], dim=1)

                end_counts = (output_ids == sep_id).sum(1)  # 统计出现的end标记
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    # 说明出现终止了～
                    # print(repeat_word)
                    # print(yayun_chars)
                    return output_ids[best_one]
                else :
                    # 保留未完成部分
                    flag = (end_counts < 1)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        token_ids = token_ids[flag]
                        token_type_ids = token_type_ids[flag]
                        last_chars = last_chars[flag]
                        yayun_chars = yayun_chars[flag]
                        new_input_ids = new_input_ids[flag]
                        new_token_type_ids = new_token_type_ids[flag]
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        end_counts = end_counts[flag]  # 扔掉已完成end计数
                        beam_size = flag.sum()  # topk相应变化
                        flag = flag.long()


            # print(repeat_word)
            # print(yayun_chars)
            return output_ids[output_scores.argmax()]


class BertSeq2seqNet(nn.Module):
    def __init__(self, model_dir, tokenizer, keep_tokens):
        super(BertSeq2seqNet, self).__init__()
        self.bert = BertModel.from_pretrained(
                model_dir,
                vocab_size=len(tokenizer._token_dict),
                keep_tokens=keep_tokens)
        self.decoder = BertLMPredictionHead(cfg, self.embeddings.word_embeddings.weight)

        logging.info("bert_model state_dict: {}".format(list(self.bert.state_dict().keys())[:10]))

    def forward(self, input_tensor, token_type_id, position_enc=None, labels=None, device="cpu"):
        ## 传入输入，位置编码，token type id ，还有句子a 和句子b的长度，注意都是传入一个batch数据
        ##  传入的几个值，在seq2seq 的batch iter 函数里面都可以返回
        input_shape = input_tensor.shape
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        ## 构建特殊的mask
        ones = torch.ones((1, 1, seq_len, seq_len), dtype=torch.float32, device=device)
        a_mask = ones.tril() # 下三角矩阵
        s_ex12 = token_type_id.unsqueeze(1).unsqueeze(2).float()
        s_ex13 = token_type_id.unsqueeze(1).unsqueeze(3).float()
        a_mask = (1.0 - s_ex12) * (1.0 - s_ex13) + s_ex13 * a_mask 

        enc_layers, _ = self.bert(input_tensor, position_ids=position_enc, token_type_ids=token_type_id, attention_mask=a_mask, 
                                    output_all_encoded_layers=True)
        squence_out = enc_layers[-1] ## 取出来最后一层输出

        predictions = self.decoder(squence_out)

        if labels is not None:
            ## 计算loss
            ## 需要构建特殊的输出mask 才能计算正确的loss
            # 预测的值不用取最后sep符号的结果 因此是到-1
            predictions = predictions[:, :-1].contiguous()
            target_mask = token_type_id[:, 1:].contiguous()
            loss = self.compute_loss(predictions, labels, target_mask)
            return predictions, loss 
        else :
            return predictions
