# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:

import click


@click.command()
@click.option('--count', default=1, help="num")
@click.option('--rate', type=float, help='rate')
@click.option('--gender', type=click.Choice(['man', 'woman']), default='man', help='select sex')
@click.option('--center', type=str, nargs=2, help='center of circle')
def hello(count, rate, gender, center):
    print("count num:", count)
    print("rate:", rate)
    print("gender:", gender)
    print("center:", center)


if __name__ == "__main__":
    hello()
