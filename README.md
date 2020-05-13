# D2A2-GAN
Discriminator-Driven Attention-Aware GAN (D2A2-GAN)<br>
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

# Requirement
* pytorch ver1.0.0以上
