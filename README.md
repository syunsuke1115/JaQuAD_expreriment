# JaQuAD（Japanese Question Answering Dataset）の精度比較実験

## 概要・背景
日本語版QAデータセットJaQuADを用いて実験を行った。
JaQuADは2022年1月に[この論文](https://arxiv.org/abs/2202.01764)で発表された日本語版のQAデータセットである。数百文字程度の文章とそれに対する質問、その回答がセットになっている。データセットはwikiperiaの記事799本から作成された合計39696組のQAのペアからなる。
JaQuADの英語版であるSQuADはこれまで色々なところで成果を上げており、韓国語版やフランス語版なども存在するが、これまで日本語版は小規模なものしかなかった。よって今後、JaQuADも様々な日本語QAタスクへの応用が期待される。

## 公式ベースラインモデルを用いた実験
公式のベースラインモデルを利用して実験を行った。特に明示されていない場合は3epochで実験を行なっている。評価値はF1を使用した。

### Lr(learning rate)
lrを変化させながら学習を行なった。lrを上げていくと急に学習がうまくいかなくなるポイントが存在した。
<img width="714" alt="スクリーンショット 2022-04-15 13 37 02" src="https://user-images.githubusercontent.com/81937075/163519818-eb3d3033-38dc-48d9-a0da-ceeb5b164eaf.png">

### 過学習
validationスコアは、epoch数を増やしてもほとんど向上が見られなかった。5epochほど回すとかなり過学習が進んでいることがわかる。

<img width="352" alt="スクリーンショット 2022-04-15 13 39 41" src="https://user-images.githubusercontent.com/81937075/163519832-14aad36b-3136-4c52-9232-f6b3b15fc2d5.png">


## モデルを変えた実験
JaQuADデータセットを使って４つのモデルをFine-tuningし精度の比較を行った。今回使用したモデルは東北大のBERT、Rinna社のRoBERTA、Cinammon社のELECTRA、BandaiNamco社のDistillBERTの四つである。
今回の実験ではBERTが最も高精度であった。また、ELECTRAは他のモデルに比べ実行時間が1/3ほどと高速であった。DistilBERTはpredictの際に異常にメモリを使ってしまいCVが測定できなかった。Lossも他の３つのモデルに比べて著しく高かった。（他が0.8~1.5程度なのに対し、2.4程度）。原因は不明。
<img width="706" alt="スクリーンショット 2022-04-15 13 41 53" src="https://user-images.githubusercontent.com/81937075/163519852-59147008-337d-4cfb-9d7c-e34b47356f52.png">


## 定性実験
BERTとELECTRAをファインチューニングしたものを使い、3つの文章を題材に5✖️3つの質問を与え定性実験を行った。
質問1-1、2-4、2-5等、**質問文の近くに回答がある問題については比較的答えやすい**ということが推測できる。
また質問2-1等、****題材文に直接使われていない文言を使った質問**はやや答えるのが難しい**ということがわかる。

### 題材1　サイバーエージェント社のインタビュー記事

[https://www.cyberagent.co.jp/corporate/message/list/detail/id=20248](https://www.co-media.jp/article/2901)

題材文
> インターネットを軸に様々な事業を展開しております。 まず、サイバーエージェントは1998年に創業し、今年16周年目という節目にあたります。元々はBtoBといわれる対企業向けのビジネスとして、インターネットの広告代理事業から始まっている会社です。創業当初から行っており今も主力事業となっているうちの一つがインターネットの広告代理店の事業です。世の中には様々な広告の媒体があると思います。テレビCMもあれば、屋外広告、電車、雑誌、新聞、ラジオなどがある中で、私たちはインターネット専業で広告代理事業を行っています。総合広告代理店はTVを中心に全て扱うと思うのですが、私たちはその中でインターネットをメインとした専業でやっています。ここがBtoBといわれる対企業向けのビジネスとしての事業内容です
> 

質問1-1：サイバーエージェントはいつ創業したの？
- 回答BERT：1998 年
- 回答ELECTRA：1998 年

質問1-2：世の中にはどんな広告媒体があるの？
- 回答BERT：様々 な 広告 の 媒体 が ある と 思い ます 。 テレビ CM も あれ ば 、 屋外 広告
- 回答ELECTRA：広告 代理 店

質問1-3：どうやって広告代理事業を行なっているの？
- 回答BERT：インターネット 専業
- 回答ELECTRA：回答なし

質問1-4：何を軸に事業を展開しているの？
- 回答BERT：[SEP] インターネット
- 回答ELECTRA：回答なし

質問1-5：サイバーエージェントは何から始まっている会社なの？
- 回答BERT：回答なし
- 回答ELECTRA：広告 代理

### 題材2　高橋是清のwikiperia
題材文
> 高橋 是清（たかはし これきよ、1854年9月19日〈嘉永7年閏7月27日〉 - 1936年〈昭和11年〉2月26日）は、明治から昭和にかけての日本の財政家、日銀総裁、政治家[1]。立憲政友会第4代総裁。第20代内閣総理大臣（在任: 1921年〈大正10年〉11月13日 - 1922年〈大正11年〉6月12日）。栄典は正二位大勲位子爵。幼名は和喜次（わきじ）。近代日本を代表する財政家として知られ、総理大臣としてよりも大蔵大臣としての評価の方が高い。愛称は「**ダルマさん**」。
> 

質問2-1：高橋 是清の職業は？
- 回答BERT：財政 家
- 回答ELECTRA：高橋 是清

質問2-2：高橋 是清は立憲政友会の第何代総裁？
- 回答BERT：第 4
- 回答ELECTRA：第 4

質問2-3：高橋 是清はなんと知られているの？
- 回答BERT：回答なし
- 回答ELECTRA：[SEP] 高橋 是清

質問2-4：高橋 是清の愛称は？
- 回答BERT：「 ダルマ さん
- 回答ELECTRA：「 ダルマ さん

質問2-5：高橋 是清の幼名は？
- 回答BERT：和 喜次
- 回答ELECTRA：和 喜次

### 題材3　サイバーエージェント社AI事業部の説明　　[https://hrmos.co/pages/cyberagent-group/jobs/0000071](https://hrmos.co/pages/cyberagent-group/jobs/0000071)
題材文
> 継続して価値を提供するためにチームとしての力を最大化して開発しようという組織風土があります。最新の情報キャッチアップできるようGoogle I/OやAWS re:Invent、国際学会などの海外で開催されるカンファレンスへの参加制度、勉強会やゼミ制度などがあり、切磋琢磨できる環境があります。最近ではサーバサイドエンジニアも対象とした機械学習ハンズオンやKaggle形式のデータサイエンスコンペティションなども実施されています。日々最新技術やマーケット情報が飛び交う環境下で技術感度の高いエンジニアと一緒にこの大きな時代の転換期にこそできる挑戦をしませんか？
> 

質問3-1：どんな環境下で挑戦できるの？
- 回答BERT：環境 下
- 回答ELECTRA：回答なし

質問3-2：最近はサーバサイドエンジニアも対象に何が行われているの？
- 回答BERT：機械 学習 ハンズオン
- 回答ELECTRA：機械 学習 ハンズオン

質問3-3：最近はサーバサイドエンジニアも対象に何が実施されているの？
- 回答BERT：機械 学習 ハンズオン
- 回答ELECTRA：機械 学習 ハンズオン 

質問3-4：どんな組織風土があるの？
- 回答BERT：回答なし
- 回答ELECTRA：回答なし

質問3-5：データサイエンスコンペティションはどんな形式？
- 回答BERT：Kaggle 形式
- 回答ELECTRA：Kaggle 形式
