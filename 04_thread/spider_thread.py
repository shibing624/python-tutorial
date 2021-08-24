import requests
from multiprocessing import Pool  # 进程池
from multiprocessing.dummy import Pool as ThreadPool  # 线程池
from logger import get_logger
logger = get_logger(__name__, 'log.txt')

def get_data_from_url(url):
    txt = requests.get(url).text
    logger.info('url:{}, size:{}'.format(url, len(txt)))
    return txt


if __name__ == '__main__':
    url_list = ['https://www.jianshu.com/p/f8c5719e5af4',
                'https://github.com/zhangjunhd/reading-notes/blob/master/literature/why-do-people-live.md',
                'https://github.com/hli1221']

    tpool = ThreadPool(20)  # 创建一个线程池，20个线程数
    data_list = tpool.map(get_data_from_url, url_list)  # 将任务交给线程池，所有url都完成后再继续执行，与python的map方法类似

    tpool.close()
    tpool.join()
    head_data = [i[:100] for i in data_list if i]

    print(len(data_list))
    print(head_data)

    print("*" * 43)
    pool = Pool(4)
    data_list = pool.map(get_data_from_url, url_list)  # 与线程池的map方法工作原理一致

    pool.close()
    pool.join()
    head_data = [i[:100] for i in data_list if i]

    print(len(data_list))
    print(head_data)
