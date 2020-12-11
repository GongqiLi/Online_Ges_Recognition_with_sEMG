from ctypes import *
import numpy as np
import torch


class SubStruct(Structure):
    _fields_ = [
        ("sub_test_int", c_int),
        ("sub_test_char_arr", c_char * 300)
    ]


class NetDvrAlarm(Structure):
    _fields_ = [
        ("test_int", c_int),
        ("char_array", c_char * 20000),
        ("test_sub_struct", SubStruct),
        ("byte_test_p", POINTER(c_byte))
    ]


CALLFUNC = CFUNCTYPE(c_void_p, POINTER(NetDvrAlarm))


def callback_msg(type_struct):
    str_tem = str(type_struct.contents.char_array.decode())
    data_tem = []
    str_list_tem = str_tem.split(',')
    str_list_tem.pop()
    int_tem_array = []
    for index, item in enumerate(str_list_tem):
        item = int(item)
        int_tem_array.append(item)
        if (index + 1) % 8 == 0:
            data_tem.append(int_tem_array)
            int_tem_array = []
    return data_tem


my_lib = windll.LoadLibrary("FetchGForceData64.dll")
my_lib.check(CALLFUNC(callback_msg), None)
