"""Helper functions, temporarily named this way."""
import sys


def ccharp(inpstr):
    """Make sure input strings are c_char_p bytes objects."""
    # https://stackoverflow.com/questions/23852311/different-behaviour-of-ctypes-c-char-p  # noqa
    if sys.version_info < (3, 0) or 'bytes' in str(type(inpstr)):
        outstr = inpstr
    else:
        outstr = inpstr.encode('utf-8')
    return outstr
