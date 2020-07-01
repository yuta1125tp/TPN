# coding:utf-8
"""eval_print()"""


from __future__ import print_function
import sys


def eval_print(expression, msg_fmt='{expression}\t{value}', 
               sep='', end='\n', file=sys.stdout, flush=False):
    """eval expression then print it with its name."""
    frame = sys._getframe(1)
    value = repr(eval(expression, frame.f_globals, frame.f_locals))

    msg = msg_fmt.format(
        expression=expression, 
        value=value)

    print(msg, sep=sep, end=end, file=file, flush=flush)


def main():
    """main func."""
    pass

if __name__=="__main__":
    main()