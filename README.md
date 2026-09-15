M1328001 駱沛葳 碩士論文

!!注意事項!!:
    1.和GNN建圖相關的套件版本不能太新，會報錯還會影響效能，請參考 environment.yml.
    2.因為兩階段的設計，所以有兩個訓練流程，另外在消融實驗中加入二階段策略的第二個模型輸入只需放活檢的類別:
        Propose         |Label
        -----------------------------------------------------------
        All dataset     |0:活檢良性，1:活檢惡性，2:切除標本
        Stage1 dataset  |0:活檢，1:切除標本
        Stage2 dataset  |0:良性，1:惡性
    3.因為兩階段的設計，所以一開始就要先切分好資料集，避免兩階段測試集混到訓練集，請參考splits_5foldtest




#主程式使用流程:
    1.前處理使用 build_tg.py ，輸入是原始影像(Origin)，輸出是.bin檔(Tissue_Graph).
    2.階段一訓練使用 train_stage1.py
    3.提取階段一的readout當作後續交叉注意力的Q，使用 extracte_emb_s1_gin.py
    4.包含交叉注意力的階段二訓練，使用 train_stage2att.py




#比較不同架構的方法實驗:
    A.CNN-based
        1.ResNet-34:輸入是裁切過的影像(Crop).
        2.EfficientNetV2-S:輸入是裁切過的影像(Crop).
        3.DenseNet-121:輸入是裁切過的影像(Crop).
        4.ConvNeXtV2-T:輸入是裁切過的影像(Crop).
        5.MSBP-Net:輸入是裁切過的影像(Crop).

    B.Vit-based
        1.VisionTransformer:輸入是裁切過的影像(Crop).
        2.SwinTransformerV2-T:輸入是裁切過的影像(Crop).
        3.SSMamba:輸入是裁切過的影像(Crop).

    C.MIL-based
        !!注意!!前處理先將原始影像切成tile，所以一個case會有不只一張影像  
        1.TransMIL:輸入是.pt檔.
        2.CLAM:輸入是.pt檔.
        3.FR-MIL:輸入是tiles，請參考 https://github.com/PhilipChicco/FRMIL
    
    D.GNN-based

        1.PatchGCN:輸入是.pt檔.
        2.CGT:輸入是.bin檔，請參考 https://github.com/wang-kang-6/CGT.git
        3.Fire-GNN:輸入是影像，請參考 https://github.com/basiralab/FireGNN
