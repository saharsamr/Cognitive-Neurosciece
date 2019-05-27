from psychopy import visual, core, monitors, event
from psychopy.visual.rect import Rect


if __name__ == "__main__":

    name = input('Name: ')

    morph_levels = [-50, -37.5, -25, -12.5, 0, 0, 12.5, 25, 37.5, 50]
    poses = [(0.0, 0.5), (-0.5, 0.0)]

    with open('result/' + name + '.csv', 'w+') as file:
        win = visual.Window(size=(2400, 1600), winType='pygame')
        Rect(win, width=1., height=1., color=[0, 0, 0])

        stim = visual.ImageStim(win)
        core.checkPygletDuringWait = False

        for pos, angle in zip(poses, [90, 180]):
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
                index = i % 10
                stim.setImage('../data/FaceGen/' + str(morph_levels[index]) + '.bmp')
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

                file.write(str(angle) + ', ' + str(morph_levels[index]) + ', ' + keys[0] + '\n')
