"""
A list of dummy players and utils for use in tests.
"""

from pathlib import Path
from typing import List


def in_docker() -> bool:
    """Return True if in running within docker, else False.

    Reference: https://stackoverflow.com/a/73564246/678486
    """
    cgroup: Path = Path("/proc/self/cgroup")
    return (
        Path("/.dockerenv").is_file()
        or cgroup.is_file()
        and "docker" in cgroup.read_text()
    )


dummy_players: List[str] = [
    "Alice",
    "Bob",
    "Carla",
    "Donald",
    "Erica",
    "Frank",
    "Gerry",
    "Harry",
    "Irina",
    "Joe",
    "Karen",
    "Larry",
    "Maureen",
    "Neil",
    "Olivia",
    "Pedro",
    "Quentin",
    "Roberta",
    "Stefan",
    "Teresa",
    "Ulfric",
    "Vicky",
    "Wendall",
    "Ximena",
    "Yan",
    "Zoe",
    "Andy",
    "Barbera",
    "Colin",
    "Deborah",
    "Edward",
    "Florence",
    "Gary",
    "Helen",
    "Ian",
    "Janet",
    "Kevin",
    "Laura",
    "Matthew",
    "Nora",
    "Oliver",
    "Patti",
    "Quincy",
    "Richard",
    "Susie",
    "Tom",
    "Ulrika",
    "Victor",
    "Wanda",
    "Xander",
    "Yasmin",
    "Zach",
    "Andrea",
    "Billy",
    "Carol",
    "Dylan",
    "Eleanor",
    "Felix",
    "Georgie",
    "Henry",
    "Isla",
    "Jonah",
    "Kylie",
    "Lachlan",
    "Mary",
    "Norbert",
    "Oksana",
    "Paul",
    "Queenie",
    "Robbie",
    "Sonia",
    "Terry",
    "Una",
    "Vishnu",
    "Waynetta",
    "Xavi",
    "Yolanda",
    "Zebedee",
]
