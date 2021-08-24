# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import smtplib
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr

import requests
from bs4 import BeautifulSoup
from lxml import etree

# 常量
from_addr = 'xx@126.com'
password = 'xx'
to_addr = ['xx@126.com', 'xx@baidu.com']
smtp_server = 'smtp.126.com'
url = 'http://wufazhuce.com/'

# 标记
isSent = False


# 编码转换方法
def _format_addr(s):
    name, addr = parseaddr(s)
    return formataddr((Header(name, 'utf-8').encode(), addr))


# 邮件方法
def sendEmail(text, img, title, story, to_addr):
    msg = MIMEMultipart()

    msg['From'] = _format_addr(u'XuMing <%s>' % from_addr)
    msg['To'] = _format_addr(u'管理员 <%s>' % to_addr)
    msg['Subject'] = Header(u'The One    ' + title, 'utf-8').encode()

    msg.attach(MIMEText('<html><body><div style="text-align: center;"><p><img src="' + img + '"></p></div>' +
                        '<p style="text-align:center;\"> <br /><br /><strong><span style="font-size:14px; text-align: center;\">' + text +
                        '</span></p><br /><br /><br /><br /><br />' + story + '</body></html>',
                        'html', 'utf-8'))

    server = smtplib.SMTP(smtp_server, 25)
    server.set_debuglevel(1)
    server.login(from_addr, password)
    server.sendmail(from_addr, [to_addr], msg.as_string())
    server.quit()


def http(url):
    html = requests.get(url).text

    page = etree.HTML(html.lower())
    # print(page)


    soup_main = BeautifulSoup(html)
    # "一个"的文字
    div = soup_main.find_all("div", {"class": "fp-one-cita"})
    text = div[0].a.text
    # print(text)

    # “一个”的图片地址
    img_list = soup_main.find_all("img", {"class": "fp-one-imagen"})
    imgUrl = img_list[0].get('src')
    # print(imgUrl)

    # "一个"的标题
    title_list = soup_main.find_all("p", {"class": "titulo"})
    title = str(title_list[0].text)
    print(title)

    # title = title.replace("VOL.","")
    # # “一个”的文章vol.1132#articulo'
    # url_stroy = 'http://wufazhuce.com/ariticle/' + title
    # # http://wufazhuce.com/article/1326

    # 得到文章的地址 用Xpath方法
    url_story = page.xpath("//*[@id=\"main-container\"]/div[1]/div[2]/div/div/div[1]/div/p[2]/a/@href")
    print(url_story[0])

    soup_stroy = BeautifulSoup(requests.get(url_story[0]).text)
    stroy_content = str(soup_stroy.find("div", {"class": "articulo-contenido"}))

    stroy_title = str(soup_stroy.find("h2", {"class": "articulo-titulo"}))

    stroy = stroy_title + stroy_content

    for addr in to_addr:
        sendEmail(text, imgUrl, title, stroy, addr)


http(url)
exit()
