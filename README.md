# D2A2-GAN
Discriminator-Driven Attention-Aware GAN (D2A2-GAN)<br>
識別器の注視領域に着目して画像を生成するGAN.<br>
GeneratorはDCGAN (Deep Convolutional GAN)とほぼ同様に構築した．<br>
Discminatorは最終層をAttention branchとAdversarial branchに分割した．<br>
Attention branchはABN (Attention Branch Network)を参考に導入しており，入力画像に対するAttention mapの生成及びクラス分類を行う．<br>
Adversarial branchは，Attention機構を用いてAttention mapを特徴マップへ反映して敵対的な誤差を出力する．

* main.py<br>
D2A2-GANを動かすためのメインのソースコード．

* utils.py<br>
Test時に生成した画像及びAttention mapをTesnorboardに書き込むためのソースコード．

* nets/Generator.py<br>
Generatorのネットワークが記述してある．

* nets/Discriminator.py<br>
Disciminatorのネットワークが記述してある．

生成した画像，Attention mapや誤差は，tensorboardに書き込むように作成している．

# Useage
* cifar10
```
python3 main.py --training_data_name cifar10 --epoch 100 --gpu 0
```

* svhn
```
python3 main.py --training_data_name svhn --epoch 100 --gpu 0
```

generating_images_for_data_augmentation.pyを動かすとクラスラベルと生成画像をディレクトリに保存する．
クラスラベルは，onehot及びAttention branchの出力にsoftmax関数を施したものを保存する．

# Requirement
* python3
* pytorch ver1.0.0以上
