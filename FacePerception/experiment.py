from psychopy import visual, core, monitors, event
from psychopy.visual.rect import Rect


if __name__ == "__main__":

    name = input('Name: ')

    win = visual.Window(size=(2400, 1600), winType='pygame')
    Rect(win, width=1., height=1., color=[0, 0, 0])

    morph_levels = [-50, -37.5, -25, -12.5, 0, 0, 12.5, 25, 37.5, 50]
    poses = [(0.0, 0.5), (-0.5, 0.0)]

    stim = visual.ImageStim(win)
    core.checkPygletDuringWait = False

    for pos in poses:
        stim.setImage('../data/FaceGen/box.png')
        stim.setPos(pos)
        stim.draw()
        win.flip()
        core.wait(1)

        text = visual.TextStim(win, text='Press -s- to start')
        text.draw()
        win.flip()
        event.waitKeys(keyList=["s"])
        core.wait(.1)

        for i in range(100):
            stim.setImage('../data/FaceGen/' + str(morph_levels[i % 10]) + '.bmp')
            stim.setPos(pos)
            stim.draw()
            win.flip()
            core.wait(.05)

            Rect(win, width=1., height=1.)
            stim.setImage('../data/FaceGen/arrows.png')
            stim.setPos((0, -.5))
            stim.setSize((.7, .7))
            stim.draw()
            win.flip()

            keys = event.waitKeys(keyList=["left", "right"])
            win.flip()
            core.wait(1)
