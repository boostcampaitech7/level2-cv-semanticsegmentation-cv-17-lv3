# Hand Bone Image Segmentation
## 1. 📖 프로젝트 소개

뼈는 우리 몸의 구조와 기능에 중요한 역할을 하며, 이를 정확히 분할하는 것은 의료 진단 및 치료 계획 수립에 필수적이다.

특히 딥러닝 기술을 활용한 뼈 분할은 인공지능 분야에서 주목받는 연구 주제 중 하나로, 질병 진단, 수술 계획 수립, 의료기기 제작, 의료 교육 등 다양한 분야에서 활용되고 있다. 본 프로젝트는 X-ray 이미지에서 사람의 뼈를 Segmentation 하는 인공지능 만드는 것을 목표로 한다. 

프로젝트 기간 : 24.11.11 ~ 24.11.28

```
부스트코스 강의 수강 및 과제 : 24.11.11 ~ 24.11.13
데이터 EDA / 데이터 전처리 / 베이스라인 모델 학습 : 24.11.14 ~ 24.11.19
데이터 증강 및 모델 성능 개선 : 24.11.20 ~ 24.11.24
후처리 / 앙상블 : 24.11.25 ~ 24.11.28
최종 자료 정리 및 문서화 : 24.11.29 ~ 24.12.02
```
<br/>

## 2.🧑‍🤝‍🧑 Team ( CV-17 : 상수와 불광)

<table>
    <tr height="160px">
        <td align="center" width="150px">
            <a href="https://github.com/Bbuterfly"><img height="110px"  src="https://avatars.githubusercontent.com/Bbuterfly"></a>
            <br/>
            <a href="https://github.com/Bbuterfly"><strong>김기수</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/sweetpotato15"><img height="110px"  src="https://avatars.githubusercontent.com/sweetpotato15"/></a>
            <br/>
            <a href="https://github.com/sweetpotato15"><strong>김유경</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/joonhyunkim1"><img height="110px"  src="https://avatars.githubusercontent.com/joonhyunkim1"/></a>
            <br/>
            <a href="https://github.com/joonhyunkim1"><strong>김준현</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/Heejin1002"><img height="110px" src="https://avatars.githubusercontent.com/Heejin1002"/></a>
            <br />
            <a href="https://github.com/Heejin1002"><strong>여희진</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/Two-Silver"><img height="110px" src="https://avatars.githubusercontent.com/Two-Silver"/></a>
            <br />
            <a href="https://github.com/Two-Silver"><strong>이은아</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/Gwonee"><img height="110px" src="https://avatars.githubusercontent.com/Gwonee"/></a>
            <br />
            <a href="https://github.com/Gwonee"><strong>정권희</strong></a>
            <br />
        </td>
</table> 

|Name|Roles|
|:----------:|:------------------------------------------------------------:|
|김기수|데이터 시각화, WandB 셋팅, 모델학습 시간 실험, Augmentation 실험, <br/> mmsegmentation 셋팅 및 코드 작성, 모델 실험 |
|김유경|EDA, Loss 실험, Augmentation 실험, smp 모델 실험, 후처리, 앙상블|
|김준현|Git, Encoder Test, Augmentation, 앙상블|
|여희진|EDA, smp model 실험, Augmentation, 앙상블|
|이은아|Scheduler/Optimzer 실험, Curriculum Learning, Yolo 실험|
|정권희|베이스라인 코드 리팩토링, Swin-Unet 구현|

</div>

wrap up 레포트 : [wrap up report](https://github.com/boostcampaitech7/level2-cv-semanticsegmentation-cv-17-lv3/blob/docs/%2351/Hand%20Bone%20Image%20Segmentation%20Report.pdf)

<br/>

## 3. 💻 프로젝트 수행 

### 3.1. 프로젝트 pipeline

> EDA 
![image](https://github.com/user-attachments/assets/571e98bf-4c48-43bb-8cac-1d4995297e7c)

모든 이미지 데이터를 살펴본 결과, 아래 특성을 확인함
- 손등 부위의 뼈가 겹치는 구조로 확인 -> 학습 시 해상도 증가, CLAHE 증강 활용
  - 특히, Trapezoid와 Pisiform에서 낮은 Dice 점수를 확인 
- 안쪽으로 회전된 손목 데이터 -> Rotate, Horizontal Flip 증강 및 Curriculum Learning 기법 활용
- X-ray 이미지로 명암 조절의 필요성 확인 -> Sharpen, Random Brightness Contrast, CLAHE, Canny 증강 활용
<br/>  

> Models

![image](https://github.com/user-attachments/assets/62140884-f2c4-4355-bc9c-48c8e39ec3a1)
위 그래프의 각 모델 해상도는 아래와 같음

(Swin-Unet : (1120, 1120), SMP : (512, 512), MMSegmentation : (1024, 1024), Ultralytics : (2048, 2048))

- Swin-Unet 구현, SMP, MMSegmentation, Ultralytics 라이브러리 이용하여 모델 및 인코더 실험 진행
- SMP 
  - Model : FPN, MAnet, PAN, Unet, Unet++, Linknet, DeepLabV3, DeepLabV3+, UperNet
  - Encoder : mobilenet_v2, efficientnet-b0, vgg19, resnet152, hrnet_w64, resnest14d
- MMSegmentation : Segformer
- Ultralytics : Yolo8, Yolo11
<br/>
<br/>

## 📈 4. 프로젝트 결과 

### 4-1. 최종 프로젝트 결과

![image](https://github.com/user-attachments/assets/9f3eb8c3-70e1-46a6-a972-bcdf69b5f119)

- UperNet : 전역적 맥락 정보와 세부 구조를 동시에 캡처하여 뼈의 복잡한 형태와 경계를 더 정확히 표현 가능
  - (1024 x 1024) 해상도 학습
- DeepLabV3+ : 세밀한 경계와 작은 뼈 구조를 더 잘 포착할 수 있음
  - (1024 x 1024) 해상도 학습 및 fold 별로 CLAHE, Horizontal Flip 증강 적용
- UNet : 손 뼈처럼 구조가 뚜렷한 영역에서 안정적이고 강력한 성능 제공
  - (1024 x 1024) 해상도 학습 및 ElasticTransform, Sharpen, Rotate, CLAHE, Horizontal Flip 증강 적용
<br/>

### 4-2. 프로젝트 실행방법

> train.py 실행
```
python code/train.py
```

> inference.py 실행
```
python code/inference.py
```

<br/>

## 5. 프로젝트 구조
프로젝트는 다음과 같은 구조로 구성되어 있음
```
📦level2-cv-semanticsegmentation-cv-17-lv3
 ┣ 📂YOLO # YOLO 모델 폴더
 ┣ 📂code 
 ┃ ┣ 📂curriculum_learning # Curriculum Learning 폴더
 ┃ ┣ 📂swin-unet # Swin-Unet 모델 폴더
 ┃ ┗ 📂utils
 ┣ 📂ensemble # 앙상블 폴더
 ┗ 📂mmsegmentation # MMSegmentation 폴더
```
<br/>

## 6. 기타사항

- 본 프로젝트에서 사용한 데이터셋은 캠프 교육용 라이선스의 가이드를 준수한다.
