import numpy
import PyQt5.QtWidgets
import PyQt5.uic
import sys
import vispy.app
import vispy.color
import vispy.scene
import vispy.visuals

vispy.app.use_app("pyqt5")

RESOLUTION = 201
SHEAR_SCALE = 3.0 / 100.0


class Window(PyQt5.QtWidgets.QMainWindow, PyQt5.uic.loadUiType("shear.ui")[0]):
    def __init__(self, parent=None):
        PyQt5.QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.canvas = vispy.scene.SceneCanvas(keys="interactive", show=True)
        self.view = self.canvas.central_widget.add_view()
        for line in range(0, 5):
            self.view.add(
                vispy.scene.visuals.Line(
                    numpy.array(
                        [
                            [0, line, 1],
                            [4, line, 1],
                        ]
                    )
                )
            )
            self.view.add(
                vispy.scene.visuals.Line(
                    numpy.array(
                        [
                            [line, 0, 1],
                            [line, 4, 1],
                        ]
                    )
                )
            )
        self.scatter = vispy.scene.visuals.Markers()
        self.scatter.set_data(
            numpy.zeros((1, 3)),
            edge_width=0,
            face_color=vispy.color.Color((1.0, 1.0, 1.0), 0.5),
            size=5,
            symbol="s",
        )
        self.view.add(self.scatter)
        self.view.camera = "turntable"
        self.view.camera.fov = 0.0
        self.verticalLayout.addWidget(self.canvas.native)
        self.shearX.valueChanged[int].connect(self.set_shear_x)
        self.shearY.valueChanged[int].connect(self.set_shear_y)
        self.shear_x = 0.0
        self.shear_y = 0.0
        self.colormap = vispy.color.colormap.MatplotlibColormap("rainbow")
        self.redraw()

    def redraw(self):
        shear = numpy.array([self.shear_x, self.shear_y, 0.0]) * SHEAR_SCALE
        o = numpy.array([0.0, 0.0, 0.0])
        x = numpy.array([1.0, 0.0, 0.0])
        y = numpy.array([0.0, 1.0, 0.0])
        t = numpy.array([0.0, 0.0, 1.0]) + shear

        mesh_x, mesh_y = numpy.meshgrid(
            numpy.linspace(0.0, 1.0 + self.shear_x * SHEAR_SCALE, RESOLUTION),
            numpy.linspace(0.0, 1.0 + self.shear_y * SHEAR_SCALE, RESOLUTION),
        )
        mesh = numpy.column_stack(
            (
                mesh_x.flatten(),
                mesh_y.flatten(),
                numpy.zeros(mesh_x.shape[0] * mesh_x.shape[1]),
            )
        )

        first_intersection = numpy.zeros(len(mesh), dtype=numpy.float64)
        first_intersection[:] = numpy.nan
        second_intersection = numpy.zeros(len(mesh), dtype=numpy.float64)
        second_intersection[:] = numpy.nan
        line = numpy.array([0.0, 0.0, 1.0])

        xy_face = numpy.cross(x, y)
        xy_face /= numpy.linalg.norm(xy_face)
        line_xy_face = numpy.dot(line, xy_face)
        if line_xy_face != 0.0:
            a = numpy.linalg.inv(numpy.vstack((x, y, xy_face)).transpose())

            # bottom
            d = (o - mesh).dot(xy_face) / line_xy_face
            coordinates = numpy.column_stack((mesh[:, 0], mesh[:, 1], d)) - o
            skewed_coordinates = coordinates @ a.transpose()
            mask = numpy.logical_and.reduce(
                (
                    skewed_coordinates[:, 0] >= 0,
                    skewed_coordinates[:, 1] >= 0,
                    skewed_coordinates[:, 0] <= 1,
                    skewed_coordinates[:, 1] <= 1,
                )
            )
            first_intersection[mask] = d[mask]

            # top
            d = (t - mesh).dot(xy_face) / line_xy_face
            coordinates = numpy.column_stack((mesh[:, 0], mesh[:, 1], d)) - t
            skewed_coordinates = coordinates @ a.transpose()
            mask = numpy.logical_and.reduce(
                (
                    skewed_coordinates[:, 0] >= 0,
                    skewed_coordinates[:, 1] >= 0,
                    skewed_coordinates[:, 0] <= 1,
                    skewed_coordinates[:, 1] <= 1,
                )
            )
            second_intersection[mask] = d[mask]

        xt_face = numpy.cross(x, t)
        xt_face /= numpy.linalg.norm(xt_face)
        line_xt_face = numpy.dot(line, xt_face)
        if line_xt_face != 0.0:
            a = numpy.linalg.inv(numpy.vstack((x, t, xt_face)).transpose())

            # front
            d = (o - mesh).dot(xt_face) / line_xt_face
            coordinates = numpy.column_stack((mesh[:, 0], mesh[:, 1], d)) - o
            skewed_coordinates = coordinates @ a.transpose()
            mask = numpy.logical_and.reduce(
                (
                    skewed_coordinates[:, 0] >= 0,
                    skewed_coordinates[:, 1] >= 0,
                    skewed_coordinates[:, 0] <= 1,
                    skewed_coordinates[:, 1] <= 1,
                )
            )
            second_intersection[mask] = d[mask]

            # back
            d = (y - mesh).dot(xt_face) / line_xt_face
            coordinates = numpy.column_stack((mesh[:, 0], mesh[:, 1], d)) - y
            skewed_coordinates = coordinates @ a.transpose()
            mask = numpy.logical_and.reduce(
                (
                    skewed_coordinates[:, 0] >= 0,
                    skewed_coordinates[:, 1] >= 0,
                    skewed_coordinates[:, 0] <= 1,
                    skewed_coordinates[:, 1] <= 1,
                )
            )
            first_intersection[mask] = d[mask]

        yt_face = numpy.cross(y, t)
        yt_face /= numpy.linalg.norm(yt_face)
        line_yt_face = numpy.dot(line, yt_face)
        if line_yt_face != 0.0:
            a = numpy.linalg.inv(numpy.vstack((y, t, yt_face)).transpose())

            # left
            d = (o - mesh).dot(yt_face) / line_yt_face
            coordinates = numpy.column_stack((mesh[:, 0], mesh[:, 1], d)) - o
            skewed_coordinates = coordinates @ a.transpose()
            mask = numpy.logical_and.reduce(
                (
                    skewed_coordinates[:, 0] >= 0,
                    skewed_coordinates[:, 1] >= 0,
                    skewed_coordinates[:, 0] <= 1,
                    skewed_coordinates[:, 1] <= 1,
                )
            )
            second_intersection[mask] = d[mask]

            # right
            d = (x - mesh).dot(yt_face) / line_yt_face
            coordinates = numpy.column_stack((mesh[:, 0], mesh[:, 1], d)) - x
            skewed_coordinates = coordinates @ a.transpose()
            mask = numpy.logical_and.reduce(
                (
                    skewed_coordinates[:, 0] >= 0,
                    skewed_coordinates[:, 1] >= 0,
                    skewed_coordinates[:, 0] <= 1,
                    skewed_coordinates[:, 1] <= 1,
                )
            )
            first_intersection[mask] = d[mask]

        mesh_mask = numpy.logical_and(
            ~numpy.isnan(first_intersection), ~numpy.isnan(second_intersection)
        )

        mesh[mesh_mask, 2] = (second_intersection - first_intersection)[mesh_mask]
        self.scatter.set_data(
            mesh,
            edge_width=0,
            face_color=self.colormap.map(mesh[:, 2]),
            size=5,
            symbol="o",
        )

    def set_shear_x(self):
        self.shear_x = self.shearX.value()
        self.redraw()

    def set_shear_y(self):
        self.shear_y = self.shearY.value()
        self.redraw()


if __name__ == "__main__":
    gui = PyQt5.QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    vispy.app.run()
