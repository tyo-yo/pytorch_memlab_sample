

# pytorch_memlab_sample

Qiitaの記事は[こちら](https://qiita.com/tyo_yo_/items/5f2c41e0dbea9276f42b)

![](https://i.imgur.com/AwnSXWN.png)



# はじめに
深層学習のコードを書いている時、GPUメモリ不足エラーが起きたことはありませんか？
でも実際どこでメモリを大量に消費しているか分からない... しょうがないからバッチサイズ減らそう...となってしまうことも多いと思います。
そこで今回はPytorchで

1. どの演算でどれくらいのGPUメモリを使用しているか
2. どのテンソル・パラメーターがどれくらいGPUメモリを使用しているか

をお手軽にプロファイリングできる[pytorch_memlab](https://github.com/Stonesjtu/pytorch_memlab)というモジュールを見つけたので、実際に使ってみようと思います。

なお、この記事は[DLHacks LT](https://deeplearning.jp/ja/hacks/) にてお話しする内容になっています。



# 使い方

まずはお手軽pipでインストール

```shell
pip install pytorch_memlab
```

### どの演算でどれくらいのGPUメモリを使用しているか

@profileというデコレータをつけると、~.pyの実行終了時にプロファイル結果を表示してくれる

```python
from pytorch_memlab import profile

class Net(nn.Module):
    def __init__(self):
        # 省略
    
    @profile
    def forward(self, x):
      	x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x 
  
# プロファイル結果
# Line  Max usage   Peak usage diff max diff peak  Line Contents
#===============================================================
#    21                                               @profile
#    22                                               def forward(self, x, labels=None):
#    23    83.04M      104.00M   57.82M   62.00M          x = self.pool(F.relu(self.conv1(x)))
#														  ...
#    26   111.31M      114.00M   23.72M   10.00M          x = F.relu(self.fc1(x))
#    27   111.47M      114.00M  168.00K    0.00B          x = F.relu(self.fc2(x))
```

*   各カラムの意味はこんな感じです。maxとかpeakとかあるのは関数が何回も繰り返し呼ばれるからです。

    *   Max usage: その行が実行された直後の（pytorchが割り当てた）最大メモリ量

    *   Peak usage: その行を実行している時にキャッシュされたメモリ量の最大値

        *   （キャッシュするメモリは最小1MBずつなので、キリのいい数字になっている）

    *   diff max: その行が実行されたことによるMax usageの変化

    *   diff peak: その行が実行されたことによるPeak usageの変化

        

*   この表の読み方はこんな感じです

    *   diffの値が大きい → その行の処理は前の行と比べたくさんメモリを使っている

    *   Max usageは演算結果＋モデルのパラメタなどによるもの。Peak usageはそれ＋計算に必要なメモリ。Peak と Maxの差が大きいほど計算を展開するための一時的なメモリが多く必要だと分かる

        *   Max usageはその性質上（forward内などでは）単調に増加しそうです

    *   Peakが最大の行 → その行の処理が一番メモリを使う

        

### どのテンソル・パラメーターがどれくらいGPUメモリを使用しているか

* MemReporterクラスにモデルを渡すことでプロファイルしてくれます。

* 訓練前にレポートすることで、モデルのアーキテクチャが使っているメモリが分かります。

```python
net = Net().cuda()
reporter = MemReporter(net)

reporter.report()

# レポート結果
# Element type                                            Size  Used MEM
# -------------------------------------------------------------------------------
# Storage on cuda:0
# fc1.weight                                      (12000, 400)    18.31M
# fc1.bias                                            (12000,)    47.00K
# fc2.weight                                       (84, 12000)     3.85M
# fc2.bias                                               (84,)   512.00B
# ...
# -------------------------------------------------------------------------------
# Total Tensors: 5823806 	Used Memory: 22.22M
# The allocated memory on cuda:0: 22.22M    
```

* 訓練後にレポートすることで、勾配や流れたデータ(x,yなど)等が使用したメモリが分かります。

```python
# ~~~~~(トレーニングのコード)~~~~~

reporter.report()

# レポート結果
# Element type                                            Size  Used MEM
# -------------------------------------------------------------------------------
# Storage on cuda:0
# fc1.weight                                      (12000, 400)    18.31M
# fc1.weight.grad                                 (12000, 400)    18.31M
# fc1.bias                                            (12000,)    47.00K 
# fc1.bias.grad                                       (12000,)    47.00K
# fc2.weight                                       (84, 12000)     3.85M
# fc2.weight.grad                                  (84, 12000)     3.85M
# fc2.bias                                               (84,)   512.00B
# fc2.bias.grad                                          (84,)   512.00B
# Tensor0                                             (12000,)    47.00K
# Tensor1                                          (84, 12000)     3.85M
# Tensor2                                                (84,)   512.00B
# -------------------------------------------------------------------------------
# Total Tensors: 17716359 	Used Memory: 67.59M

```





# 実際にCIFAR10で試してみる

[公式チュートリアルの例](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)をプロファイリングしてみます。使用したコードは[Github](https://github.com/tyo-yo/pytorch_memlab_sample/blob/master/cifar.py)においてあります。

実際のプロファイリング結果を読み解くことで、メモリがどこでどれだけ使われているかを把握することが目標です。

### モデル

畳み込み層2層、FC層3層の簡単なモデルです。ここで使用メモリの違いを顕著にするためにFC1の出力を12,000次元と大きくしてみました。

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 12000)
        self.fc2 = nn.Linear(12000, 84)
        self.fc3 = nn.Linear(84, 10)
        self.criterion = nn.CrossEntropyLoss()
        
    # ① 1行命令が実行されるたびに、占有メモリの総量がどのように変化するか追跡できる
    @profile
    def forward(self, x, labels=None):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        outputs = {'y': y}
        if labels is not None:
            outputs['loss'] = self.criterion(x, labels)
        return outputs

# ② デコレータ内の各行を追跡するので、逆伝播はやむなく関数にした。mainにデコレータをつけてもいいかも？
@profile
def backward(outputs):
    outputs['loss'].backward()
```

### 学習

トレーニングのコードもシンプルです。バッチサイズは256としました。

```python
	trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=256,
                                              shuffle=True,
                                              num_workers=2)

    net = Net().cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    reporter = MemReporter(net)
    # ③ 訓練前にレポートすることで、モデルのアーキテクチャが使っているメモリがわかる。
    reporter.report()
    print('\nStart Training\n')

    for epoch in range(1):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            outputs = net(inputs, labels)
            backward(outputs)
            optimizer.step()
    print('\nTraining Finished\n')

    # ④ 訓練後にレポートすることで、勾配などが使用したメモリがわかる。
    reporter.report()
```

### プロファイリング結果

①~④のプロファイリング結果がそれぞれ出力されます。



まずは④のモデルアーキテクチャに使われるGPUメモリを見ていきます

```
-------------------------------------------------------------------------------
Storage on cuda:0
conv2.weight                                   (16, 6, 5, 5)     9.50K
conv2.bias                                             (16,)   512.00B
fc1.weight                                      (12000, 400)    18.31M
fc1.bias                                            (12000,)    47.00K
fc2.weight                                       (84, 12000)     3.85M
fc2.bias                                               (84,)   512.00B
fc3.weight                                          (10, 84)     3.50K
fc3.bias                                               (10,)   512.00B
conv1.weight                                    (6, 3, 5, 5)     2.00K
conv1.bias                                              (6,)   512.00B
-------------------------------------------------------------------------------
Total Tensors: 5823806 	Used Memory: 22.22M
The allocated memory on cuda:0: 22.22M
-------------------------------------------------------------------------------
```

単純にモデルをGPUに置くのに22MB必要なことが分かります。また、FC1がそのうちの18MBを占めていますね。パラメータ数は何かしらの積で決まることが多いので、このように1つのレイヤーのパラメタが大多数であることもしばしば見落とされてしまいます（e.g. NLPで語彙数が30,000 次元数が500だと1,500万パラメタ）



次に③の勾配や流れたデータ(x,yなど)などのメモリも見ていきましょう。CPUとGPUそれぞれ使用メモリが分かります。

```
Element type                                            Size  Used MEM
-------------------------------------------------------------------------------
Storage on cpu
Tensor0                                      (80, 3, 32, 32)   960.00K
Tensor1                                                (80,)     1.00K
-------------------------------------------------------------------------------
Total Tensors: 245840 	Used Memory: 961.00K
```

↑ CPUにもテンソルがあります。これはデータローダーから読んで、.cuda()でGPUに送る前のものですね。



```
-------------------------------------------------------------------------------
Storage on cuda:0
Tensor2                                      (80, 3, 32, 32)   960.00K
Tensor3                                                (80,)     1.00K
Tensor4                                             (80, 10)     3.50K
Tensor5                                                 (1,)   512.00B
conv2.weight                                   (16, 6, 5, 5)     9.50K
conv2.weight.grad                              (16, 6, 5, 5)     9.50K
conv2.bias                                             (16,)   512.00B
conv2.bias.grad                                        (16,)   512.00B
fc1.weight                                      (12000, 400)    18.31M
fc1.weight.grad                                 (12000, 400)    18.31M
fc1.bias                                            (12000,)    47.00K
fc1.bias.grad                                       (12000,)    47.00K
fc2.weight                                       (84, 12000)     3.85M
fc2.weight.grad                                  (84, 12000)     3.85M
fc2.bias                                               (84,)   512.00B
fc2.bias.grad                                          (84,)   512.00B
fc3.weight                                          (10, 84)     3.50K
fc3.bias                                               (10,)   512.00B
Tensor6                                         (6, 3, 5, 5)     2.00K
Tensor7                                                 (6,)   512.00B
Tensor8                                        (16, 6, 5, 5)     9.50K
Tensor9                                                (16,)   512.00B
Tensor10                                        (12000, 400)    18.31M
Tensor11                                            (12000,)    47.00K
Tensor12                                         (84, 12000)     3.85M
Tensor13                                               (84,)   512.00B
conv1.weight                                    (6, 3, 5, 5)     2.00K
conv1.weight.grad                               (6, 3, 5, 5)     2.00K
conv1.bias                                              (6,)   512.00B
conv1.bias.grad                                         (6,)   512.00B
-------------------------------------------------------------------------------
Total Tensors: 17716359 	Used Memory: 67.59M
The allocated memory on cuda:0: 67.62M
```

↑ 次にGPUについてです。これらはモデルのパラメータの更新に関係するテンソルです。先ほど見たモデルのパラメタに加え、勾配などが加わっていることが分かります。テンソルに名前をつけていないので、それぞれ何のテンソルかはちょっと分かりにくいです。



最後に、各行で使用メモリの総量がどう推移しているかを見ていきます。

まずは①のforwardから

```
Line # Max usage   Peak usage diff max diff peak  Line Contents
===============================================================
    21                                               @profile
    22                                               def forward(self, x, labels=None):
    23    83.04M      104.00M   57.82M   62.00M          x = self.pool(F.relu(self.conv1(x)))
    24    87.59M      118.00M    4.55M   14.00M          x = self.pool(F.relu(self.conv2(x)))
    25    87.59M      104.00M    0.00B  -14.00M          x = x.view(-1, 16 * 5 * 5)
    26   111.31M      114.00M   23.72M   10.00M          x = F.relu(self.fc1(x))
    27   111.47M      114.00M  168.00K    0.00B          x = F.relu(self.fc2(x))
    28   111.48M      114.00M   10.00K    0.00B          y = self.fc3(x)
    29   111.48M      114.00M    0.00B    0.00B          outputs = {'y': y}
    30   111.48M      114.00M    0.00B    0.00B          if labels is not None:
    31   111.56M      114.00M   85.00K    0.00B              outputs['loss'] = self.criterion(x, labels)
    32   111.56M      114.00M    0.00B    0.00B          return outputs
```

ここから読み取れることとしては

*   linear1を通るとMax Usageは上がるが（line 26）、forward内でピーク値の最大はconv2の118MB（line 24）
    *   線形層の次元数をかなり大きくしたが、瞬間的なメモリ使用量はまだ畳み込み層の方が大きいということです。
    *   CNNのレイヤーでmaxとpeakの差が大きくなっているが、FCでは差は小さい。これは畳み込み層ではメモリ上に色々展開しながら計算をしなければいけないためであると考えられます。
*   何やかんや100MBちょい使っている



最後に②の逆伝播のところです。

```
Line # Max usage   Peak usage diff max diff peak  Line Contents
===============================================================
    35                                           @profile
    36                                           def backward(outputs):
    37    69.74M      148.00M    2.97M   78.00M      outputs['loss'].backward()
```

*   Peakが148MBで①と比べても最大。よって逆伝播の演算が結局一番メモリを使っている
*   Maxが70MB程度であることから、逆伝播が終わった後では70MB程度がGPUに残っている
    *   モデルサイズが22MBだから残るのは44MB程度かな、と思ったのですがそれより多い。
*   Peak - Max が80MB程度であることから、計算過程では追加で80MB程度が必要
*   本当はbackward内のどこが重いのかが知りたかったけど、そこまでは（簡単には）分からなかった



# おわりに

今回は簡単なモデルに対してGPUメモリのプロファイリングをしてみました。これくらいのシンプルなケースではあまり必要なさそうですが、複雑なモデルを書いてそれがメモリエラーになった時に

*   そもそもモデルがでかすぎるのか
*   モデルの計算過程に瞬間的にメモリが必要になるのか
*   また、瞬間的にメモリを必要としているボトルネックはどこか
*   ミスで予期せぬメモリリークが起きてしまっているのか
*   （もしくはやりとりされるテンソルが大きいのか）

などの検討材料としては大きいのではと思います。特に今までnvidia-smiしか使ってないのであれば触ってみても良いのかな、と。

公式のREADMEも簡潔で分かりやすいので興味があればぜひ見てみてください。また何か間違いやコメントなどあれば随時お願いします。