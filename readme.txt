在官方YOLOv8基础上，修复了一部分bug，进行了代码清理和整理
增加了基于CUDNN的训练与推理加速
增加了基于Torch，Numpy以及系统全局的种子设置
增加了训练和预测前的显存清理
添加了一系列的改进方法：
1注意力：BAM,FREAM,CA,CAM,CBAM,BAM,COT,DA,EA,ECA,EVA,GAM,LA,MHSA,MLCA,PA,PPA,SA,SAM,SE,SG,SK,SPA,TA
2卷积：AWSConv,CConv,DConv,DSConv,GSConv,ODConv,PConv,RepConv,SAConv,SigConv,XBNConv,DWConv
3其他：CARAFE,DWConvFpn,RepFpn,ASPP,SPPFCSPC