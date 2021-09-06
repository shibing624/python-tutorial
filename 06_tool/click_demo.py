# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: click 解析库的使用
"""

import click


@click.command()
@click.option('--count', default=1, help="num")
@click.option('--rate', type=float, default=.1, help='rate')
@click.option('--gender', type=click.Choice(['man', 'woman']), default='man', help='select sex')
@click.option('--center', type=str, nargs=2, help='center of circle')
def hello(count, rate, gender, center):
    print("count num:", count)
    print("rate:", rate)
    print("gender:", gender)
    print("center:", center)


if __name__ == "__main__":
    hello()
