https://github.com/wangshub/wechat_jump_game/wiki/Android-%E5%92%8C-iOS-%E6%93%8D%E4%BD%9C%E6%AD%A5%E9%AA%A4

主要步骤：
1.mac 终端安装：brew cask install android-platform-tools

2.手机连接PC电脑, 打开调试功能
小米手机打开 USB 调试模式，操作如下：
设置--我的设备--全部参数--MIUI版本连续点击5次--返回设置--更多设置--开发者选项--打开USB调试--打开USB调试（安全设置）
PC终端输入 adb devices ，显示如下表明设备已连接
List of devices attached
6934dc33    device

附录adb的一些命令：
adb kill-server  # 关闭服务
adb start-server  # 重启服务

3.手机登录微信，并展开到通讯录页面。
4.运行python3 main.py, 自动添加好友。

5.运行python3 robot.py，自动对刚添加的好友提问，并截图记录每个问答。
