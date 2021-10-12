# Flask教程


## 教程列表

| Notebook     |      Description      |   |
|:----------|:-------------|------:|
| [04_flask/01_Flask介绍.md](https://github.com/shibing624/python-tutorial/blob/master/04_flask/01_Flask介绍.md)  | Flask介绍 | |
| [04_flask/02_Flask模板.md](https://github.com/shibing624/python-tutorial/blob/master/04_flask/02_Flask模板.md)  | Flask模板 | |
| [04_flask/03_静态文件.md](https://github.com/shibing624/python-tutorial/blob/master/04_flask/03_静态文件.md)  | Flask静态文件 | |
| [04_flask/04_数据库.md](https://github.com/shibing624/python-tutorial/blob/master/04_flask/04_数据库.md)  | Flask数据库 | |
| [04_flask/05_模板优化.md](https://github.com/shibing624/python-tutorial/blob/master/04_flask/05_模板优化.md)  | Flask模板优化 | |
| [04_flask/06_表单.md](https://github.com/shibing624/python-tutorial/blob/master/04_flask/06_表单.md)  | Flask表单 | |
| [04_flask/07_用户认证.md](https://github.com/shibing624/python-tutorial/blob/master/04_flask/07_用户认证.md)  | 用户认证 | |
| [04_flask/08_Flask应用watchlist](https://github.com/shibing624/python-tutorial/blob/master/04_flask/watchlist)  | Flask应用示例watchlist | |



## 08_Flask应用watchlist
Demo: http://watchlist.helloflask.com

![Screenshot](https://helloflask.com/screenshots/watchlist.png)

## 安装

到指定目录：
```
$ cd 04_flask
```
安装依赖:
```
$ python -m venv env  # use `virtualenv env` for Python2, use `python3 ...` for Python3 on Linux & macOS
$ source env/bin/activate  # use `env\Scripts\activate` on Windows
$ pip install -r requirements.txt
```

生成测试数据，运行:
```
$ flask forge
$ flask run
* Running on http://127.0.0.1:5000/
```

## 新增登录账户
```
$ flask admin
```

## Reference

[Flask 入门教程](https://helloflask.com/tutorial)