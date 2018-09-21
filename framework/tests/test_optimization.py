"""
Test the optimization of transfers, generating a few simplified scenarios
and checking that the optimizer finds the expected outcome.
"""
import pytest

from ..optimization_utils import *


def test_subs():
    """
    mock squads with some players predicted some points, and
    some predicted to score zero, and check we get the right starting 11.
    """

    pass

def test_captain():
    """
    mock squad with one player predicted more points than the rest - check
    he is assigned to be captain.
    """
    pass


def test_single_transfer():
    """
    mock squad with one player predicted very low score, and potential transfers
    with higher scores, check we get the best transfer.
    """
    pass


def test_double_transfer():
    """
    mock squad with two players predicted low score, see if we get better players
    transferred in.
    """
    pass
