#include <Python.h>
#include <structmember.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <array>
#include <cmath>
#include <limits>
#include <numpy/arrayobject.h>
#include <stdexcept>
#include <vector>

static PyObject* smooth_histogram(PyObject* self, PyObject* args) {
    PyObject* raw_values;
    if (!PyArg_ParseTuple(args, "O", &raw_values)) {
        return nullptr;
    }
    try {
        if (!PyArray_Check(raw_values)) {
            throw std::runtime_error("values must be a numpy array");
        }
        auto values = reinterpret_cast<PyArrayObject*>(raw_values);
        if (PyArray_NDIM(values) != 1) {
            throw std::runtime_error("values's dimension must be 1");
        }
        if (PyArray_TYPE(values) != NPY_FLOAT64) {
            throw std::runtime_error("values's type must be float");
        }
        const auto size = PyArray_SIZE(values);
        auto minimum = std::numeric_limits<double>::infinity();
        auto maximum = -std::numeric_limits<double>::infinity();
        Py_BEGIN_ALLOW_THREADS;
        for (npy_intp index = 0; index < size; ++index) {
            const auto value = *reinterpret_cast<double*>(PyArray_GETPTR1(values, index));
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
        Py_END_ALLOW_THREADS;
        const npy_intp dimension = static_cast<npy_intp>(std::ceil(maximum - minimum + 1)) + 2;
        auto result = reinterpret_cast<PyArrayObject*>(
            PyArray_Zeros(1, &dimension, PyArray_DescrFromType(NPY_FLOAT64), 0));
        Py_BEGIN_ALLOW_THREADS;
        for (npy_intp index = 0; index < size; ++index) {
            const auto value =
                (*reinterpret_cast<double*>(PyArray_GETPTR1(values, index))) - minimum + 1.0;
            const auto value_i = std::floor(value);
            const auto value_f = value - value_i;
            (*reinterpret_cast<double*>(PyArray_GETPTR1(result, static_cast<npy_intp>(value_i)))) +=
                (1.0 - value_f);
            (*reinterpret_cast<double*>(
                PyArray_GETPTR1(result, static_cast<npy_intp>(value_i) + 1))) += value_f;
        }
        Py_END_ALLOW_THREADS;
        return reinterpret_cast<PyObject*>(result);
    } catch (const std::exception& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
    }
    return nullptr;
}

// accumulate_warped_events_square is a 2D version of smooth_histogram
static PyObject* accumulate_warped_events_square(PyObject* self, PyObject* args) {
    PyObject* raw_xs;
    PyObject* raw_ys;
    if (!PyArg_ParseTuple(args, "OO", &raw_xs, &raw_ys)) {
        return nullptr;
    }
    try {
        if (!PyArray_Check(raw_xs)) {
            throw std::runtime_error("x must be a numpy array");
        }
        auto xs = reinterpret_cast<PyArrayObject*>(raw_xs);
        if (PyArray_NDIM(xs) != 1) {
            throw std::runtime_error("x's dimension must be 1");
        }
        if (PyArray_TYPE(xs) != NPY_FLOAT64) {
            throw std::runtime_error("x's type must be float");
        }

        if (!PyArray_Check(raw_ys)) {
            throw std::runtime_error("x must be a numpy array");
        }
        auto ys = reinterpret_cast<PyArrayObject*>(raw_ys);
        if (PyArray_NDIM(ys) != 1) {
            throw std::runtime_error("y's dimension must be 1");
        }
        if (PyArray_TYPE(ys) != NPY_FLOAT64) {
            throw std::runtime_error("y's type must be float");
        }
        const auto size = PyArray_SIZE(xs);
        if (PyArray_SIZE(ys) != size) {
            throw std::runtime_error("x and y must have the same size");
        }
        auto x_minimum = std::numeric_limits<double>::infinity();
        auto y_minimum = std::numeric_limits<double>::infinity();
        auto x_maximum = -std::numeric_limits<double>::infinity();
        auto y_maximum = -std::numeric_limits<double>::infinity();
        Py_BEGIN_ALLOW_THREADS;
        for (npy_intp index = 0; index < size; ++index) {
            const auto x = *reinterpret_cast<double*>(PyArray_GETPTR1(xs, index));
            const auto y = *reinterpret_cast<double*>(PyArray_GETPTR1(ys, index));
            x_minimum = std::min(x_minimum, x);
            y_minimum = std::min(y_minimum, y);
            x_maximum = std::max(x_maximum, x);
            y_maximum = std::max(y_maximum, y);
        }
        Py_END_ALLOW_THREADS;
        const std::array<npy_intp, 2> dimensions{
            static_cast<npy_intp>(std::ceil(y_maximum - y_minimum + 1)) + 2,
            static_cast<npy_intp>(std::ceil(x_maximum - x_minimum + 1)) + 2};
        auto result = reinterpret_cast<PyArrayObject*>(
            PyArray_Zeros(2, dimensions.data(), PyArray_DescrFromType(NPY_FLOAT64), 0));
        Py_BEGIN_ALLOW_THREADS;
        for (npy_intp index = 0; index < size; ++index) {
            const auto x =
                (*reinterpret_cast<double*>(PyArray_GETPTR1(xs, index))) - x_minimum + 1.0;
            const auto y =
                (*reinterpret_cast<double*>(PyArray_GETPTR1(ys, index))) - y_minimum + 1.0;
            const auto xi = std::floor(x);
            const auto yi = std::floor(y);
            const auto xf = x - xi;
            const auto yf = y - yi;
            (*reinterpret_cast<double*>(
                PyArray_GETPTR2(result, static_cast<npy_intp>(yi), static_cast<npy_intp>(xi)))) +=
                (1.0 - xf) * (1.0 - yf);
            (*reinterpret_cast<double*>(PyArray_GETPTR2(
                result, static_cast<npy_intp>(yi), static_cast<npy_intp>(xi) + 1))) +=
                xf * (1.0 - yf);
            (*reinterpret_cast<double*>(PyArray_GETPTR2(
                result, static_cast<npy_intp>(yi) + 1, static_cast<npy_intp>(xi)))) +=
                (1.0 - xf) * yf;
            (*reinterpret_cast<double*>(PyArray_GETPTR2(
                result, static_cast<npy_intp>(yi) + 1, static_cast<npy_intp>(xi) + 1))) += xf * yf;
        }
        Py_END_ALLOW_THREADS;
        return reinterpret_cast<PyObject*>(result);
    } catch (const std::exception& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
    }
    return nullptr;
}

static PyObject* intensity_variance(PyObject* self, PyObject* args) {
    PyObject* raw_ts;
    PyObject* raw_xs;
    PyObject* raw_ys;
    double velocity_x;
    double velocity_y;
    if (!PyArg_ParseTuple(args, "OOOdd", &raw_ts, &raw_xs, &raw_ys, &velocity_x, &velocity_y)) {
        return nullptr;
    }
    try {
        if (!PyArray_Check(raw_ts)) {
            throw std::runtime_error("t must be a numpy array");
        }
        auto ts = reinterpret_cast<PyArrayObject*>(raw_ts);
        if (PyArray_NDIM(ts) != 1) {
            throw std::runtime_error("t's dimension must be 1");
        }
        if (PyArray_TYPE(ts) != NPY_FLOAT64) {
            throw std::runtime_error("t's type must be float");
        }
        if (!PyArray_Check(raw_xs)) {
            throw std::runtime_error("t must be a numpy array");
        }
        auto xs = reinterpret_cast<PyArrayObject*>(raw_xs);
        if (PyArray_NDIM(xs) != 1) {
            throw std::runtime_error("x's dimension must be 1");
        }
        if (PyArray_TYPE(xs) != NPY_FLOAT64) {
            throw std::runtime_error("x's type must be float");
        }
        if (!PyArray_Check(raw_ys)) {
            throw std::runtime_error("x must be a numpy array");
        }
        auto ys = reinterpret_cast<PyArrayObject*>(raw_ys);
        if (PyArray_NDIM(ys) != 1) {
            throw std::runtime_error("y's dimension must be 1");
        }
        if (PyArray_TYPE(ys) != NPY_FLOAT64) {
            throw std::runtime_error("y's type must be float");
        }
        const auto size = PyArray_SIZE(ts);
        if (PyArray_SIZE(xs) != size) {
            throw std::runtime_error("t and x must have the same size");
        }
        if (PyArray_SIZE(ys) != size) {
            throw std::runtime_error("t and y must have the same size");
        }
        auto variance = 0.0;
        Py_BEGIN_ALLOW_THREADS;
        auto x_minimum = std::numeric_limits<double>::infinity();
        auto y_minimum = std::numeric_limits<double>::infinity();
        auto x_maximum = -std::numeric_limits<double>::infinity();
        auto y_maximum = -std::numeric_limits<double>::infinity();
        std::vector<std::pair<double, double>> warped(size, {0.0, 0.0});
        for (npy_intp index = 0; index < size; ++index) {
            warped[index] = {
                *reinterpret_cast<double*>(PyArray_GETPTR1(xs, index))
                    - velocity_x * (*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index))),
                *reinterpret_cast<double*>(PyArray_GETPTR1(ys, index))
                    - velocity_y * (*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index)))};
            x_minimum = std::min(x_minimum, std::get<0>(warped[index]));
            y_minimum = std::min(y_minimum, std::get<1>(warped[index]));
            x_maximum = std::max(x_maximum, std::get<0>(warped[index]));
            y_maximum = std::max(y_maximum, std::get<1>(warped[index]));
        }
        const auto width = static_cast<std::size_t>(std::ceil(x_maximum - x_minimum + 1)) + 2;
        const auto height = static_cast<std::size_t>(std::ceil(y_maximum - y_minimum + 1)) + 2;
        std::vector<double> cumulative_map(width * height, 0.0);
        for (npy_intp index = 0; index < size; ++index) {
            const auto x = std::get<0>(warped[index]) - x_minimum + 1.0;
            const auto y = std::get<1>(warped[index]) - y_minimum + 1.0;
            const auto xi = std::floor(x);
            const auto yi = std::floor(y);
            const auto xf = x - xi;
            const auto yf = y - yi;
            cumulative_map[xi + yi * width] += (1.0 - xf) * (1.0 - yf);
            cumulative_map[xi + 1 + yi * width] += xf * (1.0 - yf);
            cumulative_map[xi + (yi + 1) * width] += (1.0 - xf) * yf;
            cumulative_map[xi + 1 + (yi + 1) * width] += xf * yf;
        }
        auto mean = 0.0;
        auto m2 = 0.0;
        for (std::size_t index = 0; index < cumulative_map.size(); ++index) {
            const auto delta = cumulative_map[index] - mean;
            mean += delta / static_cast<double>(index + 1);
            m2 += delta * (cumulative_map[index] - mean);
        }
        variance = m2 / static_cast<double>(cumulative_map.size());
        Py_END_ALLOW_THREADS;
        return PyFloat_FromDouble(variance);
    } catch (const std::exception& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
    }
    return nullptr;
}

static PyObject* intensity_maximum(PyObject* self, PyObject* args) {
    PyObject* raw_ts;
    PyObject* raw_xs;
    PyObject* raw_ys;
    double velocity_x;
    double velocity_y;
    if (!PyArg_ParseTuple(args, "OOOdd", &raw_ts, &raw_xs, &raw_ys, &velocity_x, &velocity_y)) {
        return nullptr;
    }
    try {
        if (!PyArray_Check(raw_ts)) {
            throw std::runtime_error("t must be a numpy array");
        }
        auto ts = reinterpret_cast<PyArrayObject*>(raw_ts);
        if (PyArray_NDIM(ts) != 1) {
            throw std::runtime_error("t's dimension must be 1");
        }
        if (PyArray_TYPE(ts) != NPY_FLOAT64) {
            throw std::runtime_error("t's type must be float");
        }
        if (!PyArray_Check(raw_xs)) {
            throw std::runtime_error("t must be a numpy array");
        }
        auto xs = reinterpret_cast<PyArrayObject*>(raw_xs);
        if (PyArray_NDIM(xs) != 1) {
            throw std::runtime_error("x's dimension must be 1");
        }
        if (PyArray_TYPE(xs) != NPY_FLOAT64) {
            throw std::runtime_error("x's type must be float");
        }
        if (!PyArray_Check(raw_ys)) {
            throw std::runtime_error("x must be a numpy array");
        }
        auto ys = reinterpret_cast<PyArrayObject*>(raw_ys);
        if (PyArray_NDIM(ys) != 1) {
            throw std::runtime_error("y's dimension must be 1");
        }
        if (PyArray_TYPE(ys) != NPY_FLOAT64) {
            throw std::runtime_error("y's type must be float");
        }
        const auto size = PyArray_SIZE(ts);
        if (PyArray_SIZE(xs) != size) {
            throw std::runtime_error("t and x must have the same size");
        }
        if (PyArray_SIZE(ys) != size) {
            throw std::runtime_error("t and y must have the same size");
        }
        auto maximum = 0.0;
        Py_BEGIN_ALLOW_THREADS;
        auto x_minimum = std::numeric_limits<double>::infinity();
        auto y_minimum = std::numeric_limits<double>::infinity();
        auto x_maximum = -std::numeric_limits<double>::infinity();
        auto y_maximum = -std::numeric_limits<double>::infinity();
        std::vector<std::pair<double, double>> warped(size, {0.0, 0.0});
        for (npy_intp index = 0; index < size; ++index) {
            warped[index] = {
                *reinterpret_cast<double*>(PyArray_GETPTR1(xs, index))
                    - velocity_x * (*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index))),
                *reinterpret_cast<double*>(PyArray_GETPTR1(ys, index))
                    - velocity_y * (*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index)))};
            x_minimum = std::min(x_minimum, std::get<0>(warped[index]));
            y_minimum = std::min(y_minimum, std::get<1>(warped[index]));
            x_maximum = std::max(x_maximum, std::get<0>(warped[index]));
            y_maximum = std::max(y_maximum, std::get<1>(warped[index]));
        }
        const auto width = static_cast<std::size_t>(std::ceil(x_maximum - x_minimum + 1)) + 2;
        const auto height = static_cast<std::size_t>(std::ceil(y_maximum - y_minimum + 1)) + 2;
        std::vector<double> cumulative_map(width * height, 0.0);
        for (npy_intp index = 0; index < size; ++index) {
            const auto x = std::get<0>(warped[index]) - x_minimum + 1.0;
            const auto y = std::get<1>(warped[index]) - y_minimum + 1.0;
            const auto xi = std::floor(x);
            const auto yi = std::floor(y);
            const auto xf = x - xi;
            const auto yf = y - yi;
            cumulative_map[xi + yi * width] += (1.0 - xf) * (1.0 - yf);
            maximum = std::max(maximum, cumulative_map[xi + yi * width]);
            cumulative_map[xi + 1 + yi * width] += xf * (1.0 - yf);
            maximum = std::max(maximum, cumulative_map[xi + 1 + yi * width]);
            cumulative_map[xi + (yi + 1) * width] += (1.0 - xf) * yf;
            maximum = std::max(maximum, cumulative_map[xi + (yi + 1) * width]);
            cumulative_map[xi + 1 + (yi + 1) * width] += xf * yf;
            maximum = std::max(maximum, cumulative_map[xi + 1 + (yi + 1) * width]);
        }
        Py_END_ALLOW_THREADS;
        return PyFloat_FromDouble(maximum);
    } catch (const std::exception& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
    }
    return nullptr;
}

static PyMethodDef event_warping_extension_methods[] = {
    {"smooth_histogram", smooth_histogram, METH_VARARGS, nullptr},
    {"accumulate_warped_events_square", accumulate_warped_events_square, METH_VARARGS, nullptr},
    {"intensity_variance", intensity_variance, METH_VARARGS, nullptr},
    {"intensity_maximum", intensity_maximum, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};
static struct PyModuleDef event_warping_extension_definition = {
    PyModuleDef_HEAD_INIT,
    "event_warping_extension",
    "event_warping_extension speeds up some sessiontools operations",
    -1,
    event_warping_extension_methods};
PyMODINIT_FUNC PyInit_event_warping_extension() {
    auto module = PyModule_Create(&event_warping_extension_definition);
    import_array();
    return module;
}
