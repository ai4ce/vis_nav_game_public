from .interface import *

try:
    import vis_nav_game.core
except:
    print("vis_nav_game.core.*.pyd might not be imported properly!")


def play(the_player: Player) -> bool:
    """
    This is the main entry point for players to play the vis_nav_game. Players should define their own
    Player class by inheriting from Player in the vis_nav_game.interface
    :param the_player: a subclass of Player
    :return: True if the game is successfully ran, False if the_player did not pass the test
    """
    return vis_nav_game.core.play(the_player)