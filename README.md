# yolov8_rknn_Cplusplus

yolov8 瑞芯微 rknn 板端 C++部署，使用平台 rk3588。模型转换参考 [onnx转rknn](https://blog.csdn.net/zhangqian_1/article/details/128918268) ， 仿真参考[PC仿真](https://github.com/cqu20160901/yolov8n_onnx_tensorRT_rknn_horizon) 。

## 编译和运行

1）编译

```
cd examples/rknn_yolov8_demo_open

bash build-linux_RK3588.sh

```

2）运行

```
cd install/rknn_yolov8_demo_Linux

./rknn_yolov8_demo

```

注意：修改模型、测试图像、保存图像的路径，修改文件为src下的main.cc

```

int main(int argc, char **argv)
{
    char model_path[256] = "/home/zhangqian/rknn/rknpu2_1.4.0/examples/rknn_yolov8_demo_open/model/RK3588/yolov8n_ZQ.rknn";
    char image_path[256] = "/home/zhangqian/rknn/rknpu2_1.4.0/examples/rknn_yolov8_demo_open/test.jpg";
    char save_image_path[256] = "/home/zhangqian/rknn/rknpu2_1.4.0/examples/rknn_yolov8_demo_open/test_result.jpg";

    detect(model_path, image_path, save_image_path);
    return 0;
}
```


# 测试效果


冒号“:”前的数子是coco的80类对应的类别，后面的浮点数是目标得分。（类别:得分）

![images](https://github.com/cqu20160901/yolov8_rknn_Cplusplus/blob/main/examples/rknn_yolov8_demo_open/test_result.jpg)

（注：图片来源coco128）

说明：推理测试预处理没有考虑等比率缩放，激活函数 SiLU 用 Relu 进行了替换。由于使用的是coco128的128张图片数据进行训练的，且迭代的次数不多，效果并不是很好，仅供测试流程用。换其他图片测试检测不到属于正常现象，最好选择coco128中的图像进行测试。

把板端模型推理和后处理时耗也附上，供参考，使用的芯片rk3588。
![image](https://github.com/cqu20160901/yolov8_rknn_Cplusplus/assets/22290931/9e13c2b9-b666-45c6-bdb5-340253a69e95)


2024年1月12日对后处理进行了优化，后处理时耗有所降低。
![image](https://github.com/cqu20160901/yolov8_rknn_Cplusplus/assets/22290931/69ca1aa4-88e5-49e4-915e-3fa84384e90d)


# 相关参考链接

[【yolov8 瑞芯微RKNN和地平线Horizon芯片仿真测试部署】](https://blog.csdn.net/zhangqian_1/article/details/128918268)

[【yolov8 瑞芯微 RKNN 的 C++部署】](https://blog.csdn.net/zhangqian_1/article/details/131130085)
