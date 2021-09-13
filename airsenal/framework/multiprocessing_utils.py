"""
Custom Queue class and counter, to allow us to get the length of a queue,
which in turn lets us do the tree-based optimization.

Taken from
https://gist.github.com/FanchenBao/d8577599c46eab1238a81857bb7277c9
by Fanchen Bao, based on this Stack Overflow thread:
https://stackoverflow.com/questions/41952413/get-length-of-queue-in-pythons-multiprocessing-library
"""

from multiprocessing.queues import Queue
import multiprocessing
import os


def set_multiprocessing_start_method(num_thread=2):
    """To fix change of default behaviour in multiprocessing on Python 3.8 and later
    on MacOS. Python 3.8 and later start processess using spawn by default, see:
    https://docs.python.org/3.8/library/multiprocessing.html#contexts-and-start-methods

    Note that this should be called at most once, ideally protecteed within
    if __name__  == "__main__"

    Parameters
    ----------
    num_thread : int, optional
        Only changem ultiprocessing start method if num_thread > 1, by default 2
    """
    if num_thread is not None and num_thread > 1 and os.name == "posix":
        multiprocessing.set_start_method("fork")


# The following implementation of custom MyQueue to avoid NotImplementedError
# when calling queue.qsize() in MacOS X comes almost entirely from this github
# discussion: https://github.com/keras-team/autokeras/issues/368
# Necessary modification is made to make the code compatible with Python3.


class SharedCounter(object):
    """
    A synchronized shared counter.
    The locking done by multiprocessing.Value ensures that only a single
    process or thread may read or write the in-memory ctypes object. However,
    in order to do n += 1, Python performs a read followed by a write, so a
    second process may read the old value before the new one is written by the
    first process. The solution is to use a multiprocessing.Lock to guarantee
    the atomicity of the modifications to Value.
    This class comes almost entirely from Eli Bendersky's blog:
    http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing/
    """

    def __init__(self, n=0):
        self.count = multiprocessing.Value("i", n)

    def increment(self, n=1):
        """Increment the counter by n (default = 1)"""
        with self.count.get_lock():
            self.count.value += n

    @property
    def value(self):
        """Return the value of the counter"""
        return self.count.value


class CustomQueue(Queue):
    """
    A portable implementation of multiprocessing.Queue.
    Because of multithreading / multiprocessing semantics, Queue.qsize() may
    raise the NotImplementedError exception on Unix platforms like Mac OS X
    where sem_getvalue() is not implemented. This subclass addresses this
    problem by using a synchronized shared counter (initialized to zero) and
    increasing / decreasing its value every time the put() and get() methods
    are called, respectively. This not only prevents NotImplementedError from
    being raised, but also allows us to implement a reliable version of both
    qsize() and empty().
    """

    def __init__(self):
        super().__init__(ctx=multiprocessing.get_context())
        self.size = SharedCounter(0)

    def put(self, *args, **kwargs):
        self.size.increment(1)
        super().put(*args, **kwargs)

    def get(self, *args, **kwargs):
        self.size.increment(-1)
        return super().get(*args, **kwargs)

    def qsize(self):
        """Reliable implementation of multiprocessing.Queue.qsize()"""
        return self.size.value

    def empty(self):
        """Reliable implementation of multiprocessing.Queue.empty()"""
        return not self.qsize()
