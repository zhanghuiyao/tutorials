

### 搭建ascend+mindspore开发环境需要三步
  1. 查ascend driver/firmware/cann与mindspore配套表
  2. 下载并安装ascend driver/firmware/cann toolkit
  3. 下载并安装mindspore
#### 第一步 查ascend driver/firmare/cann与mindspore配套表

| mindspore  | ascend driver | firmware    | cann toolkit/kernel
| :---       |:---           | :--         |:--
| 2.3.1      | 24.1.RC2      | 7.3.0.1.231 | 8.0.RC2.beta1
| 2.3.0      | 24.1.RC2      | 7.3.0.1.231 | 8.0.RC2.beta1
| 2.2.10     | 23.0.3        | 7.1.0.5.220 | 7.0.0.beta1
| 2.1.0      | 23.0.rc2      | 6.4.12.1.241 | 6.3.RC2

如要安装ms 2.3.1，ascend driver/firmware/cann相应的版本就是 24.1.RC2/7.3.0.1.231/8.0.RC2.beta1
- 温馨提示1: 请按照配套关系下载安装对应的版本，很多算法开发遇到的问题都来自于环境配置安装，请一定参考以上的配套关系表。
- 温馨提示2: 从硬件适配度和特性方面考虑，建议使用最新的2.3.0/2.3.1版本进行算法开发，有问题请到gitee提issue，https://gitee.com/mindspore/mindspore/issues

#### 第二步 下载并安装ascend driver/firmware/cann toolkit
 - ascend的driver/cann软件有两套版本系列，一个是商用版本，一个是社区版本。以下例子均来自于社区版本。两者区别与版本对应关系稍后整理

仍然以安装mindspore 2.3.1和2.3.0为例，硬件以atlas 800T A2训练服务器 aarch64 架构为例，下面表格是对应的版本run包下载方式

序号  | ascend software             | version | package name | download | release date| 
|:-- |:---                  |:---    |:---     | :--- | :---
1    |ascend driver         | 24.1.RC2 | Ascend-hdk-910b-npu-driver_24.1.rc2_linux-aarch64.run| [download link](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend%20HDK/Ascend%20HDK%2024.1.RC2/Ascend-hdk-910b-npu-driver_24.1.rc2_linux-aarch64.run?response-content-type=application/octet-stream) | 2024-07-31 |
2    |ascend firmware       | 7.3.0.1.231 | Ascend-hdk-910b-npu-firmware_7.3.0.1.231.run | [download link](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend%20HDK/Ascend%20HDK%2024.1.RC2/Ascend-hdk-910b-npu-firmware_7.3.0.1.231.run?response-content-type=application/octet-stream) | 2024-07-31
3    |cann toolkit          | 8.0.RC2.beta1 | Ascend-cann-toolkit_8.0.RC2_linux-aarch64.run | [download page](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC2.beta1)| 2024-07-17
4    |cann kernel           |  8.0.RC2.beta1  | Ascend-cann-kernels-910b_8.0.RC2_linux.run | [download page](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC2.beta1) | 2024-07-17 |

- driver和firmware的download link放的是run包下载地址，点击即可下载，cann toolkit和 kernel放的是download page，需要进入页面后，注册登陆账号才可下载。
- 想了解更多ascend cann信息，请查看https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC2.beta1

##### 1. 安装driver和firmware
```bash
# 安装driver 
./Ascend-hdk-910b-npu-driver_24.1.rc2_linux-aarch64.run --full --install-for-all

# 查看NPU卡信息
npu-smi info

# 查看安装driver的版本号，显示24.1.rc2
cat /usr/local/Ascend/driver/version.info

# 安装NPU firmware
./Ascend-hdk-910b-npu-firmware_7.3.0.1.231.run --full

# 查看安装firmware的版本号，显示 7.3.0.1.231
cat /usr/local/Ascend/firmware/version.info


# 重启OS
reboot
```

##### 2. 安装cann
```bash

# 安装 cann toolkit
./Ascend-cann-toolkit_8.0.RC2_linux-aarch64.run --install --install-for-all --quiet

# 安装二进制算子包cann-kernel
./Ascend-cann-kernels-910b_8.0.RC2_linux.run --install --install-for-all --quiet

# 然后执行如下命令配置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 查看cann 版本号
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg  
# 将会显示7.3.0.1.231:8.0.RC2 -> 7.3.0.1.231是firmware版本号, 8.0.RC2是cann tooklit版本号
```
##### 3. 安装sympy/hccl/te
```bash
pip install sympy
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/te-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/hccl-*-py3-none-any.whl
```

#### 第三步 下载并安装mindspore
mindspore的whl包有两种安装方式
1. 直接pip install mindspore==2.3.1 or pip install mindspore==2.3.0（自动根据系统python版本和cpu架构安装相应的whl包） 
2. 手动选择python版本和cpu架构相对应的whl包，下载和安装。
##### mindspore 2.3.1 
| python  | os    | cpu     | mindspore whl  | 
| :---:    |:---:   |:---:  | :--- |
| 3.8     | linux | aarch64 | [mindspore-2.3.1-cp38-cp38-linux_aarch64.whl](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.3.1/MindSpore/unified/aarch64/mindspore-2.3.1-cp38-cp38-linux_aarch64.whl)
| 3.9     | linux | aarch64 | [mindspore-2.3.1-cp39-cp39-linux_aarch64.whl](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.3.1/MindSpore/unified/aarch64/mindspore-2.3.1-cp39-cp39-linux_aarch64.whl)
| 3.10     | linux | aarch64 | [mindspore-2.3.1-cp310-cp310-linux_aarch64.whl](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.3.1/MindSpore/unified/aarch64/mindspore-2.3.1-cp310-cp310-linux_aarch64.whl)
| 3.8     | linux | x86_64 | [mindspore-2.3.1-cp38-cp38-linux_x86_64.whl](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.3.1/MindSpore/unified/x86_64/mindspore-2.3.1-cp38-cp38-linux_x86_64.whl)
| 3.9     | linux | x86_64 | [mindspore-2.3.1-cp39-cp39-linux_x86_64.whl](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.3.1/MindSpore/unified/x86_64/mindspore-2.3.1-cp39-cp39-linux_x86_64.whl)
| 3.10     | linux | x86_64 | [mindspore-2.3.1-cp310-cp310-linux_x86_64.whl](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.3.1/MindSpore/unified/x86_64/mindspore-2.3.1-cp310-cp310-linux_x86_64.whl)
##### mindspore 2.3.0 
| python  | os    | cpu     | mindspore whl  | 
| :---:    |:---:   |:---:   | :--- |
| 3.8     | linux | aarch64 | [mindspore-2.3.0-cp38-cp38-linux_aarch64.whl](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.3.0/MindSpore/unified/aarch64/mindspore-2.3.0-cp38-cp38-linux_aarch64.whl)
| 3.9     | linux | aarch64 | [mindspore-2.3.0-cp39-cp39-linux_aarch64.whl](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.3.0/MindSpore/unified/aarch64/mindspore-2.3.0-cp39-cp39-linux_aarch64.whl)
| 3.10     | linux | aarch64 | [mindspore-2.3.0-cp310-cp310-linux_aarch64.whl](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.3.0/MindSpore/unified/aarch64/mindspore-2.3.0-cp310-cp310-linux_aarch64.whl)
| 3.8     | linux | x86_64 | [mindspore-2.3.0-cp38-cp38-linux_x86_64.whl](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.3.0/MindSpore/unified/x86_64/mindspore-2.3.0-cp38-cp38-linux_x86_64.whl)
| 3.9     | linux | x86_64 | [mindspore-2.3.0-cp39-cp39-linux_x86_64.whl](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.3.0/MindSpore/unified/x86_64/mindspore-2.3.0-cp39-cp39-linux_x86_64.whl)
| 3.10     | linux | x86_64 | [mindspore-2.3.0-cp310-cp310-linux_x86_64.whl](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.3.0/MindSpore/unified/x86_64/mindspore-2.3.0-cp310-cp310-linux_x86_64.whl)



```bash
# 查看mindspore版本号
python -c "import mindspore;mindspore.set_context(device_target='Ascend');mindspore.run_check()"
# 将会显示MindSpore version: 2.3.0或者2.3.1
```


