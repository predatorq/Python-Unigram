"""
Helper function for handling UTF8 bytes. 
"""

from functools import lru_cache


@lru_cache()
def bytes_to_unicode():
    """
    Lookup tables between utf-8 bytes and unicode strings.

    From: <https://github.com/openai/gpt-2/blob/master/src/encoder.py>
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))
