# yolov8_rknn_Cplusplus
yolov8 瑞芯微 rknn 板端 C++部署，使用平台 rk3588。模型转换参考 [onnx转rknn](https://github.com/cqu20160901/yolov8n_onnx_tensorRT_rknn_horizon) ， 仿真参考[PC仿真](https://github.com/cqu20160901/yolov8n_onnx_tensorRT_rknn_horizon) 。

# 部署测试效果

冒号“:”前的数子是coco的80类对应的类别，后面的浮点数是目标得分。（类别:得分）

![images](https://github.com/cqu20160901/yolov8_rknn_Cplusplus/blob/main/examples/rknn_yolov8_demo_open/test_result.jpg)

（注：图片来源coco128）

说明：推理测试预处理没有考虑等比率缩放，激活函数 SiLU 用 Relu 进行了替换。由于使用的是coco128的128张图片数据进行训练的，且迭代的次数不多，效果并不是很好，仅供测试流程用。换其他图片测试检测不到属于正常现象，最后选择coco128中的图像进行训练。

# 相关参考链接
[【yolov8 瑞芯微RKNN和地平线Horizon芯片仿真测试部署】](https://blog.csdn.net/zhangqian_1/article/details/128918268)
[【yolov8 瑞芯微 RKNN 的 C++部署】](https://blog.csdn.net/zhangqian_1/article/details/131130085)
