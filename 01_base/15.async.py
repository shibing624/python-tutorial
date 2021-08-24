# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""


def demo4():
    """
    这是最终我们想要的实现.
    """
    import asyncio  # 引入 asyncio 库

    async def washing1():
        await asyncio.sleep(3)  # 使用 asyncio.sleep(), 它返回的是一个可等待的对象
        print('washer1 finished')

    async def washing2():
        await asyncio.sleep(2)
        print('washer2 finished')

    async def washing3():
        await asyncio.sleep(5)
        print('washer3 finished')

    """
    事件循环机制分为以下几步骤:
        1. 创建一个事件循环
        2. 将异步函数加入事件队列
        3. 执行事件队列, 直到最晚的一个事件被处理完毕后结束
        4. 最后建议用 close() 方法关闭事件循环, 以彻底清理 loop 对象防止误用
    """
    # 1. 创建一个事件循环
    loop = asyncio.get_event_loop()

    # 2. 将异步函数加入事件队列
    tasks = [
        washing1(),
        washing2(),
        washing3(),
    ]

    # 3. 执行事件队列, 直到最晚的一个事件被处理完毕后结束
    loop.run_until_complete(asyncio.wait(tasks))
    """
    PS: 如果不满意想要 "多洗几遍", 可以多写几句:
        loop.run_until_complete(asyncio.wait(tasks))
        loop.run_until_complete(asyncio.wait(tasks))
        loop.run_until_complete(asyncio.wait(tasks))
        ...
    """

    # 4. 如果不再使用 loop, 建议养成良好关闭的习惯
    # (有点类似于文件读写结束时的 close() 操作)
    loop.close()

    """
    最终的打印效果:
        washer2 finished
        washer1 finished
        washer3 finished
        elapsed time = 5.126561641693115
        	(毕竟切换线程也要有点耗时的)

    说句题外话, 我看有的博主的加入事件队列是这样写的:
        tasks = [
            loop.create_task(washing1()),
            loop.create_task(washing2()),
            loop.create_task(washing3()),
        ]
        运行的效果是一样的, 暂不清楚为什么他们这样做.
    """


if __name__ == '__main__':
    # 为验证是否真的缩短了时间, 我们计个时
    from time import time
    start = time()

    # demo1()  # 需花费10秒
    # demo2()  # 会报错: RuntimeWarning: coroutine ... was never awaited
    # demo3()  # 会报错: RuntimeWarning: coroutine ... was never awaited
    demo4()  # 需花费5秒多一点点

    end = time()
    print('elapsed time = ' + str(end - start))