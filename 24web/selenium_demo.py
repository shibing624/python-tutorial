# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: selenium用于爬虫，主要是用来解决javascript渲染的问题
"""

import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait


def search_baidu_and_fetch(url='https://www.baidu.com', query='姚明老婆是谁？'):
    browser.get(url)
    q = browser.find_element_by_id('kw')
    q.send_keys(query)
    q.send_keys(Keys.ENTER)
    wait = WebDriverWait(browser, 3)
    wait.until(expected_conditions.presence_of_element_located((By.ID, 'content_left')))
    print('current_url:', browser.current_url)
    print('get_cookies:', browser.get_cookies())
    print('page_source:', browser.page_source[:100])
    time.sleep(1)


def get_page_source():
    url = 'https://www.baidu.cn'
    browser.get(url)
    print('url:{}, page_source:{}'.format(url, browser.page_source[:100]))


def get_page_element():
    browser.get('http://www.taobao.com')
    print(browser.page_source)
    lst = browser.find_element_by_css_selector('li')
    lst_c = browser.find_element(By.CSS_SELECTOR, 'li')
    print(lst, lst_c)


def get_page_search_element():
    """对获取到的元素调用交互方法"""
    browser.get('https://www.baidu.com')
    q = browser.find_element_by_id('kw')
    q.send_keys('iphone')
    q.send_keys(Keys.ENTER)
    print(browser.current_url)
    print(len(browser.page_source))
    time.sleep(5)
    q.clear()
    q.send_keys('ipad')
    q.send_keys(Keys.ENTER)
    # button = browser.find_element_by_class_name('btn-search')
    # button.click()
    print(browser.current_url)
    print(len(browser.page_source))
    time.sleep(5)


def add_action_source():
    from selenium.webdriver import ActionChains

    url = 'https://www.runoob.com/try/try.php?filename=jqueryui-api-droppable'
    browser.get(url)
    browser.switch_to.frame('iframeResult')
    source = browser.find_element_by_id('draggable')
    target = browser.find_element_by_id('droppable')
    actions = ActionChains(browser)
    actions.drag_and_drop(source, target)
    actions.perform()
    '''
    1.先用switch_to_alert()方法切换到alert弹出框上
    2.可以用text方法获取弹出的文本 信息
    3.accept()点击确认按钮
    4.dismiss()相当于点右上角x，取消弹出框
    '''
    time.sleep(2)
    print(browser.current_url)


def exe_script():
    browser.get('https://www.zhihu.com/explore')
    browser.execute_script('window.scrollTo(0,document.body.scrollHeight)')
    browser.execute_script('alert("To button")')


def get_text():
    browser.get('https://www.zhihu.com/explore')
    logo = browser.find_element_by_id("Popover1-toggle")
    print(logo)
    print(logo.text)
    print(logo.get_attribute("class"))
    print('logo id, location, tag_name, size:')
    print(logo.id, logo.location, logo.tag_name, logo.size)


if __name__ == '__main__':
    # ps：启动环境要求：1.打开safari的偏好设置你的高级-开发菜单；2.开发菜单中打开允许远程自动化。
    browser = webdriver.Safari()

    # search_baidu_and_fetch()
    # get_page_search_element()
    # add_action_source()
    # exe_script()
    get_text()
    browser.close()
