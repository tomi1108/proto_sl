# プロトタイプを用いたSplit Learning
- 使用モデル：MobileNet
- データセット：CIFAR-10
- クライアント数：2
- Non-IID：クラス0~4と5~9に分けて、各クライアントに分配
- 損失：class lossとproto lossを足し合わせて計算（エポックごとに足し合わせる割合が変化していく）