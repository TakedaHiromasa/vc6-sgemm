# vc6-sgemm
py-videocore6を使ったSGEMM実装練習  

## py-videocore6 リポジトリ
https://github.com/Idein/py-videocore6  

## Installation
必ず`py-videocore6`のリポジトリを確認すべし．  

## SGEMM 概要
```
1 パラメータ読み込み 
2 アドレス初期化
3 Mの分のループ
    4 Nの分のループ
        5 読み出しアドレス計算 
        6 Kの分のループ
            7 直積計算
      8 計算結果転送
9 終了処理
```