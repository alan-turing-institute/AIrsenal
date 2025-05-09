from typing import List

from tqdm import tqdm

from airsenal.framework.optimization_transfers import activate_assistant_manager
from airsenal.framework.optimization_utils import get_num_increments, get_starting_squad
from airsenal.framework.schema import session
from airsenal.framework.squad import Squad
from airsenal.framework.utils import (
    get_latest_prediction_tag,
    get_player,
)


def main(
    squad: Squad,
    gw_range: List[int],
    season: str,
    tag: str,
    num_iter=100,
    dbsession=session,
) -> None:
    best_squad = None
    best_transfers = {}
    best_pts = 0
    for nt in [3]:  # [0, 1, 2, 3]:
        num_increments_for_updater = 20 * get_num_increments(nt, num_iter)
        pid = f"A{nt}"
        progress_bar = tqdm(total=num_increments_for_updater, desc=pid)
        increment = 1

        def updater(increment=1, index=None):
            progress_bar.update(increment)

        try:
            new_squad, new_transfers, new_pts = activate_assistant_manager(
                nt,
                squad,
                tag,
                gw_range,
                gw_range[0],
                season,
                num_iter=num_iter,
                update_func_and_args=(updater, increment, pid),
                verbose=False,
            )
            print("Points", new_pts)
            print(new_squad)
            print_transfers(new_transfers)
            if new_pts > best_pts:
                best_squad = new_squad
                best_transfers = new_transfers
                best_pts = new_pts
        except Exception as e:
            print(f"Failed with {nt}: {e}")

    print("==========================")
    print("Best squad:")
    print("Points:", best_pts)
    print(best_squad)
    print_transfers(best_transfers)


def print_transfers(transfers):
    print("Transfers:")
    print(f"- IN: {[get_player(player_id).name for player_id in transfers['in']]}")
    print(f"- OUT: {[get_player(player_id).name for player_id in transfers['out']]}")


if __name__ == "__main__":
    squad = get_starting_squad()
    gw_range = [36, 37, 38]
    season = "2425"
    tag = get_latest_prediction_tag()
    main(squad=squad, gw_range=gw_range, season=season, tag=tag)
