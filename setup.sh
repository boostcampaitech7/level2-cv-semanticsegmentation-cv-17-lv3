#!/bin/bash

pip install --upgrade pip # pip 패키지 업데이트
apt update # 패키지 목록 업데이트
apt install wget # wget 다운로드
apt install git # git 다운로드

# url initialization
CODE_URL='https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000338/data/20241107093156/code.tar.gz'
DEPENDENCY_PATH='code/requirements.txt'

# code download
wget "$CODE_URL"
tar -zxvf code.tar.gz

rm -rf *.tar.gz

pip install -r "$DEPENDENCY_PATH"