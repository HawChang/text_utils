#!/usr/bin/env python
# -*- coding:gbk -*-
"""
Author:   zhanghao55@baidu.com
Date  :   18/08/15 14:46:20
Desc  :   ͨ�ó�����, ��ˮ�س���
"""
import random

class Sampler(object):
    """ͨ�ó����� ��ˮ�س���
    """
    
    def __init__(self, sample_num):
        """��ʼ�� ָ����������
        Args:
            sample_num : ָ����������
        """
        assert isinstance(sample_num, int), "sample_num must be integer."
        self._sample_num = sample_num
        self._sample_list = list()
        self._count = 0

    def put(self, obj):
        """
        Args:
            obj : ����Ԫ��
        """
        if len(self._sample_list) < self._sample_num:
            self._sample_list.append(obj)
        else:
            index = random.randint(0, self._count)
            if index < self._sample_num:
                self._sample_list[index] = obj
        self._count += 1

    def get_sample_list(self):
        """���ص�ǰ�������
        """
        return self._sample_list
    
    def clear(self):
        """��ճ����б�
        """
        self._sample_list[:]=[]
        self._count = 0


def test():
    """���Ի�������
    """
    sampler = Sampler(10)
    for i in range(100):
        sampler.put(i)
    print(sampler.get_sample_list())

    tests = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    sampler = Sampler(4)
    for i in range(10):
        sampler.clear()
        for num in tests:
            sampler.put(num)
        print(sampler.get_sample_list())

if __name__ == '__main__': 
    test()
