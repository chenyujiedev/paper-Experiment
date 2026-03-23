# paper-실험

- kr
- Zero-DCE_code_mac -> Myloss_light_mac.py에는 핵심인 DarkRegionExposureLoss 및 HighlightPreservationLoss 코드(논문에 소개된 손실 함수 알고리즘)가 포함되어 있습니다.
- Image_xxx_xxx 두 파일 패키지는 각각 NIQE, BRISQUE, LOE, PSMR, SSIM을 통계 및 계산합니다.
- Retinexformer와 RUAS는 공식 소스 코드로, Zero-DCE와 Retinex에 포함된 데이터셋을 불러와 훈련을 수행했습니다.
- Zero-DCE_code_mac은 논문의 주요 훈련 코드이며, lowlight_train_mac.py에는 저조도 실험을 위한 파라미터 코드가 포함되어 있습니다.
- cn
- Myloss_light_mac.py有核心的DarkRegionExposureLoss和HighlightPreservationLoss代码（出现的论文中的loss算法）
- Image_xxx_xxx 两个文件包分别是统计和计算NIQE、BRISQUE、LOE、PSMR、SSIM.
- Retinexformer 和 RUAS 是官方源码，就导入了Zero-DCE和 retinex自带的数据集进行训练.
- Zero-DCE_code_mac 为论文 主要训练代码，lowlight_train_mac.py中暗区实验参数代码.
