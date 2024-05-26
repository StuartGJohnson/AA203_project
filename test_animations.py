import unittest
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class animated_class():
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.t = np.linspace(0, 3, 40)
        g = -9.81
        v0 = 12
        self.z = g * self.t ** 2 / 2 + v0 * self.t

        v02 = 5
        self.z2 = g * self.t ** 2 / 2 + v02 * self.t

        self.scat = self.ax.scatter(self.t[0], self.z[0], c="b", s=5, label=f'v0 = {v0} m/s')
        self.line2 = self.ax.plot(self.t[0], self.z2[0], label=f'v0 = {v02} m/s')[0]
        self.ax.set(xlim=[0, 3], ylim=[-4, 10], xlabel='Time [s]', ylabel='Z [m]')
        self.ax.legend()

    def update(self, frame):
        # for each frame, update the data stored on each artist.
        x = self.t[:frame]
        y = self.z[:frame]
        # update the scatter plot:
        data = np.stack([x, y]).T
        self.scat.set_offsets(data)
        # update the line plot:
        self.line2.set_xdata(self.t[:frame])
        self.line2.set_ydata(self.z2[:frame])
        return self.scat, self.line2

class MyTestCase(unittest.TestCase):
    def test_something(self):
        ac = animated_class()
        ani = animation.FuncAnimation(fig=ac.fig, func=ac.update, frames=40, interval=30)
        ani.save(filename="ffmpeg_example.mp4", writer="ffmpeg")
        #self.assertEqual(True, False)  # add assertion here

if __name__ == '__main__':
    unittest.main()
