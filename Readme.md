### 環境

作業系統 : WSL2 Ubuntu 20.04.06 LTS

Python 3.12

### 安裝套件

```bash
    ./install.sh
```

### 執行程式

```bash
    ./run.sh
```

### 更改訓練圈數

./src/main.py : 20
```python=
    train( X_train , y_train , NowDateTime , batch_size = 128 , epochs = 次數 )
```

Output.csv 會在同層
丟給天氣小幫手前需要手動把 upload.csv 和 output.csv 的第一行（中文）刪除
