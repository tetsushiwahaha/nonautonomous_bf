# nonautonomous_bf
IEICE Fundamentals Review「Pythonで挑む分岐解析」の付属プログラムです．

## 動作環境
* Python3
* numpy, scipy

## 使用法
func_gen.pyに系を記述した後：
```
python3 func_gen.py
python3 bf.py <your_input_json_file>
```

## 入力ファイル
* "xrange", "yrange" : x/y軸の描画範囲
* "x0" : 初期値
* "params" : パラメタ
* "dparams" : 変分パラメタの変化量
* "btype" : 分岐の種類の指定{"G", "I", "NS"}
* "errors" : Newton法の収束判定
* "explosion" : Newton法の発散判定の上限値
* "period" : 指定周期ごとに周期時刻を計算
* "ite_max" : Newton法の最大繰り返し数
* "increment_param" : 変分パラメタを指定
* "variable_param" : 変数パラメタを指定

## 使用例
"func_gen.py"に系を記述し，系のヤコビアン等のコードを自動生成します．("sys_func.py"が生成されます)．
その後，付属のNS分岐用の入力ファイルを読ませます．
```
python3 func_gen.py 
python3 bf.py inNS.json
```
標準出力に次のような出力がされます．
```
3  0.05000  0.05000  0.01393  1.00000 -1.53503784  0.13093856  0.57996900 C  0.2256802  0.9742014 ABS  1.0000000 (41[ms])
```
これは次の情報に相当します．
```
<Newton法収束回数>  <パラメタ> <固定点座標> <実虚数判定 R or C>  <特性定数> <特性定数の絶対値> (<実行時間>)
```
