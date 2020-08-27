#!/usr/bin/env python
# -*- coding: gb18030 -*-
 
"""
Author:   zhanghao55@baidu.com
Date  :   19/09/19 21:12:03
DESC  :   通用切词类
"""

import jieba
import word_seg

class WordSegger(object):
    """通用切词类
    """
    def __init__(self, seg_method="word_seg", segdict_path="src/text_utils/dict/chinese_gbk"):
        """切词类初始化
        [in] seg_method: str, 指明切词方式 jieba 还是内部切词工具
             segdict_path: str, 内部切词工具需要切词字典地址
        """
        self.seg_words = {
            "jieba": self.jieba_seg_words,
            "word_seg": self.baidu_seg_words
        }[seg_method]

        self._segger = None
        if seg_method == "word_seg":
            self._segger = word_seg.WordSeg(segdict_path)
            self._segger.init_wordseg_handle()

    def baidu_seg_words(self, text):
        """使用公司内部切词工具
        [in]  text: str, 待切词字符串， unicode或gb18030编码
        [out] seg_list: list[str], 切词结果，unicode编码
        """
        if isinstance(text, unicode):
            text = text.encode("gb18030", "ignore")
        # 内部切词函数接受gbk编码字符串
        return [x.decode("gb18030") for x in self._segger.seg_words(text)]

    def jieba_seg_words(self, text):
        """使用结巴进行分词
        [in]  text: str, 待切词字符串，unicode或gb18030编码
        [out] seg_list: list[str], 切词结果，unicode编码
        """
        if isinstance(text, unicode):
            text = text.encode("gb18030", "ignore")
        # jieba分词结果是unicode编码
        return [x for x in jieba.lcut(text)]
    
    def destroy(self):
        """内部切词工具需要释放内存
        """
        if self._segger is not None:
            self._segger.destroy_wordseg_handle()


if __name__ == "__main__":
    # 测试
    segger = WordSegger(segdict_path="./dict/chinese_gbk")
    print(" ".join(segger.seg_words("测试该切词类处理gb18030编码的字符串")).encode("gb18030"))
    print(" ".join(segger.seg_words(u"再看看unicode编码的字符串是否也可以")).encode("gb18030"))
    print(" ".join(segger.seg_words(u"孩子近视，度数六百度，去哪配眼镜？")).encode("gb18030"))

    line = u"icl做手术需要多少钱 500度能不能服兵役吗 做近视恢复icl做手术多少钱 公务员体检视力小窍门 近视icl做手术哪里做 icl晶体植入完后可以开车 近视眼治疗费用 icl做治疗近视那个医院好 icl近视的危害 近视激光治疗费用 近视眼手术治疗费用 近视眼icl做手术多少钱啊 icl晶体植入后多久跑步 近视icl做手术费用要多少 飞秒治近视手术 icl做手术治疗近视多少钱 眼科专家在线咨询 500度一定不能考空乘吗 怎么治疗近视 眼底黄斑如何治疗近视怎么考公务员,咨询郑州华厦眼科医院 近视怎么考公务员,哪家眼科医院做近视手术好?郑州华厦眼科医院引进美国新 一代i-fs飞秒设备,助您放心摘镜. 近视眼的治疗方法,全飞和半飞哪种手术方式好 近视眼的治疗方法,郑州华厦眼科医院,医生团队是河南省优势学科. 术后视力可达1.0,更有患者视力达到1.2,值得信任. 近视术后注意事项 近视需要注意些什么? 近视术后注意事项专业眼科医生在线解析 治疗近视选择郑州华厦医院 河南叁甲眼科医院 河南飞秒近视手术第壹人 十余年品牌,采用美国飞秒系统治疗近视,请您放心. 郑州治疗近视多少费用 19年统一标准收费-郑州华厦 郑州治疗近视多少费用咨询郑州华厦医院 河南叁甲眼科专科医院 医保定点单位 激光治疗近视眼选郑州华厦医院,还您清晰世界. 郑州眼科医院 原郑州艾格眼科医院 郑州眼科医院哪家好?选郑州华厦 省级专业眼科医院,眼科博士医生团队, 河南人自己的专业眼科医院 进口眼科检查手术设备, 咨询电话:0371-60100601 高度近视做手术可靠吗?有后遗症吗? 高度近视做手术 选郑州华厦 徐惠民亲自坐诊 经验丰富 近视手术规范成熟 采用可靠 的icl植入手术,升级21项术前检查,一觉醒来还你清晰视力 近视手术价格 2019近视手术价格一览 近视手术价格是多少?咨询郑州华厦医院 医保定点医院 在线讲解不同近视手术方式的 收费标准和手术要求. 全飞秒近视有副作用吗郑州眼科国家医院专业眼科 全飞秒近视有副作用吗?郑州华厦眼科在线详解 三级专科医院 医保定点 国际新进技术 .专业医生团队 飞秒手术治疗近视30秒快速摘镜 21道术前检查 还您清晰视力. 眼睛近视怎么矫正_三种矫正近视的好方法? 眼睛近视怎么矫正,郑州哪家眼科医院近视手术好?2019近视好评眼科医院-郑州华厦 眼科医院.点击了解. 怎样矫正近视_美国新一代i-fs飞秒设备助您摘镜 怎样矫正近视,咨询郑州华厦眼科医院,2019河南近视飞秒手术好评眼科医院. 专业眼科医生为您放心矫正视力. 医生讲解 近视手术哪里好?__问医生 答 : 郑州华厦眼科 近视手术安全吗?郑州华厦眼科多种近视 矫治方法,采用美国近视矫治近视,全程无刀矫治,让您享受清晰世界. 近视手术费用 郑州做近视手术多少钱? 近视手术费用?2019郑州近视手术哪家眼科医院收费合理?在线点击详情了解 激光治疗近视手术,术后视力可达到多少? 激光治疗近视手术哪种方式比较好?郑州华厦眼科医院,医生团队是河南省优势学科, 经验丰富.术后视力可达1.0,更有患者视力达到1.2,值得信任 近视医生河南眼科博士 完成近视手术5万余例 近视医生,郑州华厦眼科医院,十余年品牌.近视手术采用美国ifs飞秒设备,助您放心 摘镜 得了近视眼怎么办 近视眼如何有效医治? 得了近视眼怎么办咨询郑州华厦医院,十余年品牌.近视手术采用美国ifs飞秒设备 .助您放心摘镜 医生讲解 近视手术安荃可靠吗?__问医生 答 : 近视手术 咨询郑州华厦医院 郑州近视专科医院,华厦眼科多种近视 矫治方法,采用美国近视矫治近视,全程无刀矫治,让您享受清晰世界. 郑州眼科医院 郑州新农合指定医院 郑州华厦眼科医院是省级专业眼科医院,眼科博士医生团队,河南人自己的专业眼科医院 先进进口眼科检查手术设备,成为河南眼病患者自己的 咨询电话:0371-60100601 飞秒手术治疗近视原理,在线咨询医生.飞秒手术治疗近视原理,郑州华厦眼科医院近视飞秒手术网络预约立减3000, 医生团队是河南省优势学科.近视原理在线解析. 近视医生,郑州地区2019近视手术好评医院 近视医生,郑州近视手术治疗中心,郑州华厦眼科.十余年品牌,经验 丰富.术后视力可达1.0,更有患者视力达到1.2,近视手术在线咨询 全飞秒近视手术价格 郑州地区近视手术多少钱? 全飞秒近视手术价格?咨询 郑州华厦眼科 河南三级眼科专科医院 医保定点单位 激光治疗近视眼选 郑州华厦眼科,快速30秒,还您清晰世界.电话:免费在线咨询"

    print("line length = %d" % len(line))
    seg_line = "".join(segger.seg_words(line))
    print("seg length = %d" % len(seg_line))
    print(line.encode("gb18030"))
    print("="*150)
    print(seg_line.encode("gb18030"))

    print(" ".join(segger.seg_words(u"登基失败后被假圣旨赐死，史上最悲催的太子是谁？")).encode("gb18030"))

    segger.destroy()
