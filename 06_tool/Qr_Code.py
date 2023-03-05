'''
@jiangmiemie（jiangyangcreate@gmail）
生成二维码/解析二维码、生成动态二维码的模块
'''

from easyqr import easyqr as qr# 解析模块
#上传图片
path = r'code\ME.png'#使用你自己的微信二维码截图即可
url = qr.upload(path)
#获得解析的地址
url =qr.online(url)

from MyQR import myqr #动态二维码模块，结果不支持中文
myqr.run(
    words= str(url)          ,  # 扫描二维码后，显示的内容，或是跳转的链接
    version=9                ,  # 设置容错率
    level='L'                ,  # 控制纠错水平，范围是L、M、Q、H，从左到右依次升高
    picture='gif.gif' ,  # 图片所在目录，可以是动图
    colorized=True           ,  # 黑白(False)还是彩色(True)
    contrast=1.0             ,  # 用以调节图片的对比度，1.0 表示原始图片。默认为1.0。
    brightness=1.0           ,  # 用来调节图片的亮度，用法同上。
    save_name='xxxxx.gif'        ,  # 控制输出文件名，格式可以是 .jpg， .png ，.bmp ，.gif
    )

import qrcode #静态二维码模块，支持中文
img = qrcode.make('DESKTOP')# 填写你想要扫码出现的内容（文字/链接）
img.save('DESKTOP.png') # 填写文件保存路径