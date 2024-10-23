from vis_nav_game.interface import Player

# the following several lines of code hide the class from python's dir method,
# allowing us to conceal the ground truth information from the players.
# Note: it must be declared before including the server.py
# ref: https://stackoverflow.com/q/24466671
cdef object KeyboardPlayerPyBullet
cdef object Game
cdef object pybullet_data
cdef object PB
cdef object capture_fpv
cdef object Fernet
cdef object gdown
cdef object zipfile
cdef object hashlib
cdef object argparse
cdef object nopub

include "../vis_nav_core.py"

def play(the_player: Player) -> bool:

    # add the test run below to prevent Players from calling pybullet functions in their code
    # if a Player calls any pybullet functions, such as getBasePositionAndOrientation, then
    # the following code will generate errors because no pybullet connection has been made,
    # if the Player tries to make pybullet connections in their own code, Good luck with that!
    try:
        test_img = np.random.randint(0,255,(240,320,3), dtype=np.uint8)
        the_player.pre_exploration()
        the_player.pre_navigation()
        the_player.set_target_images([])
        the_player.see(test_img)
        the_player.act()
    except:
        import inspect
        import traceback
        print('Your Player code (attached below) did not pass the test run. Please contact us for debug:')
        print(inspect.getsource(the_player.__class__))
        print('')
        traceback.print_exc()
        return False

    the_player.reset()
    print('Your Player code pass the test run. Now begin the game!')
    the_sim = Game(player=the_player, do_pybullet_gui=False)
    the_sim.run()
    print('Game finished!')
    return True
