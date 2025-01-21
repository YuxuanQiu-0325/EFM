1. 运行训练代码train_oc20
   每训练1个epoch就保存1个模型到checkpoint_1000，eval可以指定验证的epoch数
2. 验证过程是通过加载预训练模型并且令eval=1
   例如测试epoch=150时的模型效果，则加载epoch=150的预训练模型，并且设置eval=1
   此时模型会对val_data中的lmdb文件进行验证（inpaint后保留mol用于后续测试rmsd和mae）【可将train_data清空，这样验证完之后代码就不会继续运行】
   验证结束后会保存下来json和traj文件，其中test_rmsd用来计算json文件中的rmsd，test_mae用来计算json文件中的mae
3. 验证过程有时候没办法对1个完整的lmdb文件进行验证，会有1个样本出现离谱的情况。此时需要用脚本“拆分lmdb”将1个lmdb拆解成很多个样本，并把几个样本放进val_data进行分批次验证
