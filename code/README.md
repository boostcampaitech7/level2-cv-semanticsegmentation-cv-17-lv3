## 💻 프로젝트 Summary

> 데이터셋
 
- Train : 800, Test : 288 (Total 1,088)
- 데이터셋은 크게 손가락, 손등, 팔로 구성되며 총 29개의 뼈 종류가 존재
- Input : 각 사람 마다 두 장의 양손 hand bone x-ray 객체가 담긴 이미지, segmentation annotation이 담긴 json file
- Output : 각 클래스(29개)에 대한 확률 맵을 갖는 multi-class 예측 수행하고, 해당 결과를 RLE 형식으로 변환
<br/>

> Preprocessing 
 
- 이미지 해상도 별 모델 성능 확인
  - 해상도가 높을 수록 좋은 성능을 보임
  
| **Resolution** | **Model** | **Augmentation** | **Val.Dice** | **LB Dice** |
|:-----------:|:---------:|:------------:|:----------:|:----------:|
| 512 x 512 | DeepLabV3+(resnet50) | | 0.9536 | 0.9500 |
| 1024 x 1024 | DeepLabV3+(resnet152) | |0.9690 | 0.9658 |
| 1536 x 1536 | DeepLabV3+(resnet152) | CLAHE | 0.9723 | 0.9689 |
| 512 x 512 | Unet++(resnet50) | | 0.9459 | 0.9318 | 
| 1024 x 1024 | Unet++(resnet50) | | 0.9651 | 0.9614 |
<br/>

> Augmentation

- 데이터 증강을 통한 모델 성능 확인
  - CLAHE, Horizontal Flip 적용 시 좋은 성능을 보임
  
| **Augmentation** | **Description** | **Improvement** |
|:-----------:|:---------:|:------------:|
| CLAHE | 이미지의 대비를 향상시켜 디테일을 강조 | +0.0048 |
| Rotate | 이미지 일정 각도 회전 | -0.0002 |
| Horizontal Flip | 이미지 좌우 대칭 | +0.0084 | 
| Sharpen | 커널 연산을 사용하여 픽셀간의 차이를 강조하여 선명도를 높임 | -0.0016 |
| Random Brightness Contrast | 이미지의 밝기와 대비를 무작위로 조정 | -0.0377 |
| Canny | 이미지의 윤곽선을 감지하여 경계선 강조 | -0.0146 |
<br/>


> Curriculum Learning
- 데이터 증강 : 손목 회전 데이터에 HorizonFlip 증강 기법을 적용하여 데이터를 추가 생성
- 단계적 학습
  - 첫 번째 단계 : 원본 데이터만 사용하여 모델의 기본 학습 진행
  - 두 번째 단계 : 원본 학습 모델을 기반으로, HorizonFlip 증강된 손목 회전 데이터만을 추가 학습
- 손목 회전 데이터에 대한 성능 개선에 효과적
  
| | **적용 전 LB Dice** | **적용 후 LB Dice** |
|:-----------:|:---------:|:------------:|
| LB Dice Score | 0.9147 | 0.9412 |
<br/>

> 후처리 (Negative Sample Masking)

- 추론 과정이후 29개의 class가 각각 가장 면적이 큰 하나의 Contour로 결과가 나오기 위해 후처리 진행
- 소폭 성능 개선 확인
  
| **Model** | **적용 전 LB Dice** | **적용 후 LB Dice** | **변경된 row 비율** |
|:-----------:|:---------:|:------------:|:----------:|
| DeepLabV3Plus(resnet152) | 0.9658 | 0.9660 (+0.0002) | 8% |
| UperNet(hrnet_w64) | 0.9688 | 0.9692 (+0.0004) | 3.4% |
<br/>

> Ensemble

- 성능이 가장 높았던 모델들을 활용하여, Soft Voting, Hard Voting, Classwise Ensemble 기법 적용

| **Method** | **Model1** | **Model2** | **Model3** | **LB Dice** |
|:-----------:|:---------:|:------------:|:----------:|:----------:|
| Soft Voting | DeepLabV3+(resnet152)* | UperNet(hrnet-w64) | Unet(eff-b0) | 0.9709 |
| Soft Voting | DeepLabV3+(resnet152)* | UperNet(hrnet-w64) | | 0.9707 |
| Soft Voting | Segformer(fold0) | Segformer(fold1) | Segformer(fold2) | 0.9688 |

(* : DeepLabV3+(resnet152) 모델 Fold Ensemble)
<br/>
<br/>

