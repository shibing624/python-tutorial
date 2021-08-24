# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from wxpy import *
# 初始化机器人，扫码登陆
bot = Bot()

# 搜索名称含有 "游否" 的男性深圳好友
my_friend = bot.friends().search('lili')

print(my_friend)


# 机器人账号自身
myself = bot.self

# 向文件传输助手发送消息
bot.file_helper.send('Hello from wxpy!test')

# 在 Web 微信中把自己加为好友
# bot.self.add()
# bot.self.accept()

# 发送消息给自己
# bot.self.send('能收到吗？')

# 搜索名称包含 'wxpy'，且成员中包含 `游否` 的群聊对象
wxpy_groups = bot.groups().search('家庭0.0')

print(wxpy_groups)


my_friend = ensure_one(bot.search('张'))
file_helper = bot.file_helper
file_helper.send("hello xm")
tuling = Tuling(api_key='')

my_friend.send("hello")
# 使用图灵机器人自动与指定好友聊天
@bot.register(my_friend)
def reply_my_friend(msg):
    print(msg)
    reply = tuling.do_reply(msg)
    print(reply)

@bot.register(file_helper)
def reply_file_helper(msg):
    print(msg)
    reply = tuling.do_reply(msg)
    print(reply)

embed()