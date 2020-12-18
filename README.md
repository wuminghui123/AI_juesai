
## 环境
tensorflow1.15 ascend910  

## 文件说明
链接：https://pan.baidu.com/s/1K1cqIyAWIA1X7fdDTPN0FA   
提取码：thlx   
通过上述百度网盘下载模型文件model_8.best.pb和model_32.best.pb并放置于pb_model文件夹   


请将Y_1.csv和Y_2.csv放置于data文件夹下，这里由于数据集过大不上传了，你们拷贝进去  

X_pre_1.bin和X_pre_2.bin为生成提交数据，也在data文件夹中，已经生成好，运行run.sh会再次生成提交数据

outputMessage.txt记录了控制台运行此脚本的输出信息。  

pb_model文件夹存储了8导频和32导频模型的pb图模型。  

## 运行
运行run.sh即可自动配置ascend910的npu环境变量并启动eval.py进行推理。  

大约两到三分钟后将测试集共2万个样本推理完成，推理时间两个各占30s共一分钟，剩余一分钟用在读取数据和模型初始化上。  

