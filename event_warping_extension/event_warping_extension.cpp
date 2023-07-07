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

// accumulate is a 2D version of smooth_histogram
static PyObject* accumulate(PyObject* self, PyObject* args) {
    int32_t sensor_width;
    int32_t sensor_height;
    PyObject* raw_ts;
    PyObject* raw_xs;
    PyObject* raw_ys;
    double velocity_x;
    double velocity_y;
    if (!PyArg_ParseTuple(
            args,
            "iiOOOdd",
            &sensor_width,
            &sensor_height,
            &raw_ts,
            &raw_xs,
            &raw_ys,
            &velocity_x,
            &velocity_y)) {
        return nullptr;
    }
    try {
        if (sensor_width <= 0) {
            throw std::runtime_error("sensor_width must be larger than zero");
        }
        if (sensor_height <= 0) {
            throw std::runtime_error("sensor_height must be larger than zero");
        }
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
        const auto t0 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, 0));
        const auto t1 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, size - 1));
        const auto maximum_delta_x = std::floor(std::abs(velocity_x) * (t1 - t0)) + 2.0;
        const auto maximum_delta_y = std::floor(std::abs(velocity_y) * (t1 - t0)) + 2.0;
        const auto width = static_cast<std::size_t>(sensor_width + maximum_delta_x);
        const auto height = static_cast<std::size_t>(sensor_height + maximum_delta_y);
        const std::array<npy_intp, 2> dimensions{
            static_cast<npy_intp>(height), static_cast<npy_intp>(width)};
        auto result = reinterpret_cast<PyArrayObject*>(
            PyArray_Zeros(2, dimensions.data(), PyArray_DescrFromType(NPY_FLOAT64), 0));
        Py_BEGIN_ALLOW_THREADS;
        for (npy_intp index = 0; index < size; ++index) {
            const auto warped_x =
                *reinterpret_cast<double*>(PyArray_GETPTR1(xs, index))
                - velocity_x * ((*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index))) - t0);
            const auto warped_y =
                *reinterpret_cast<double*>(PyArray_GETPTR1(ys, index))
                - velocity_y * ((*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index))) - t0);
            const auto x = warped_x + (velocity_x > 0 ? maximum_delta_x : 0.0);
            const auto y = warped_y + (velocity_y > 0 ? maximum_delta_y : 0.0);
            auto xi = std::floor(x);
            auto yi = std::floor(y);
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
    int32_t sensor_width;
    int32_t sensor_height;
    PyObject* raw_ts;
    PyObject* raw_xs;
    PyObject* raw_ys;
    double velocity_x;
    double velocity_y;
    if (!PyArg_ParseTuple(
            args,
            "iiOOOdd",
            &sensor_width,
            &sensor_height,
            &raw_ts,
            &raw_xs,
            &raw_ys,
            &velocity_x,
            &velocity_y)) {
        return nullptr;
    }
    try {
        if (sensor_width <= 0) {
            throw std::runtime_error("sensor_width must be larger than zero");
        }
        if (sensor_height <= 0) {
            throw std::runtime_error("sensor_height must be larger than zero");
        }
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
        const auto t0 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, 0));
        const auto t1 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, size - 1));
        const auto maximum_delta_x = std::floor(std::abs(velocity_x) * (t1 - t0)) + 2.0;
        const auto maximum_delta_y = std::floor(std::abs(velocity_y) * (t1 - t0)) + 2.0;
        const auto width  = static_cast<std::size_t>(sensor_width + maximum_delta_x);
        const auto height = static_cast<std::size_t>(sensor_height + maximum_delta_y);
        std::vector<double> cumulative_map(width * height, 0.0);
        for (npy_intp index = 0; index < size; ++index) {
            const auto warped_x =
                *reinterpret_cast<double*>(PyArray_GETPTR1(xs, index))
                - velocity_x * ((*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index))) - t0);
            const auto warped_y =
                *reinterpret_cast<double*>(PyArray_GETPTR1(ys, index))
                - velocity_y * ((*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index))) - t0);
            const auto x = warped_x + (velocity_x > 0 ? maximum_delta_x : 0.0);
            const auto y = warped_y + (velocity_y > 0 ? maximum_delta_y : 0.0);
            auto xi = std::floor(x);
            auto yi = std::floor(y);
            const auto xf = x - xi;
            const auto yf = y - yi;
            cumulative_map[xi + yi * width] += (1.0 - xf) * (1.0 - yf);
            cumulative_map[xi + 1 + yi * width] += xf * (1.0 - yf);
            cumulative_map[xi + (yi + 1) * width] += (1.0 - xf) * yf;
            cumulative_map[xi + 1 + (yi + 1) * width] += xf * yf;
        }
        auto mean = 0.0;
        auto m2 = 0.0;
        auto minimum_determinant = 0.0;
        auto maximum_determinant = 0.0;
        auto corrected_velocity_x = 0.0;
        auto corrected_velocity_y = 0.0;
        if ((velocity_x >= 0.0) == (velocity_y >= 0.0)) {
            corrected_velocity_x = std::abs(velocity_x);
            corrected_velocity_y = std::abs(velocity_y);
            minimum_determinant = -corrected_velocity_y * sensor_width;
            maximum_determinant = corrected_velocity_x * sensor_height;
        } else {
            corrected_velocity_x = std::abs(velocity_x);
            corrected_velocity_y = -std::abs(velocity_y);
            minimum_determinant = corrected_velocity_x * maximum_delta_y;
            maximum_determinant = corrected_velocity_x * (maximum_delta_y + sensor_height)
                                  - corrected_velocity_y * sensor_width;
        }
        std::size_t count = 0;
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const auto determinant = y * corrected_velocity_x - x * corrected_velocity_y;
                if (determinant >= minimum_determinant && determinant <= maximum_determinant) {
                    const auto value = cumulative_map[x + y * width];
                    const auto delta = value - mean;
                    mean += delta / static_cast<double>(count + 1);
                    m2 += delta * (value - mean);
                    ++count;
                }
            }
        }
        variance = m2 / static_cast<double>(count);
        Py_END_ALLOW_THREADS;
        return PyFloat_FromDouble(variance);
    } catch (const std::exception& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
    }
    return nullptr;
}



static PyObject* intensity_maximum(PyObject* self, PyObject* args) {
    int32_t sensor_width;
    int32_t sensor_height;
    PyObject* raw_ts;
    PyObject* raw_xs;
    PyObject* raw_ys;
    double velocity_x;
    double velocity_y;
    if (!PyArg_ParseTuple(
            args,
            "iiOOOdd",
            &sensor_width,
            &sensor_height,
            &raw_ts,
            &raw_xs,
            &raw_ys,
            &velocity_x,
            &velocity_y)) {
        return nullptr;
    }
    try {
        if (sensor_width <= 0) {
            throw std::runtime_error("sensor_width must be larger than zero");
        }
        if (sensor_height <= 0) {
            throw std::runtime_error("sensor_height must be larger than zero");
        }
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
        const auto t0 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, 0));
        const auto t1 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, size - 1));
        const auto maximum_delta_x = std::floor(std::abs(velocity_x) * (t1 - t0)) + 2.0;
        const auto maximum_delta_y = std::floor(std::abs(velocity_y) * (t1 - t0)) + 2.0;
        const auto width = static_cast<std::size_t>(sensor_width + maximum_delta_x);
        const auto height = static_cast<std::size_t>(sensor_height + maximum_delta_y);
        std::vector<double> cumulative_map(width * height, 0.0);
        for (npy_intp index = 0; index < size; ++index) {
            const auto warped_x =
                *reinterpret_cast<double*>(PyArray_GETPTR1(xs, index))
                - velocity_x * ((*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index))) - t0);
            const auto warped_y =
                *reinterpret_cast<double*>(PyArray_GETPTR1(ys, index))
                - velocity_y * ((*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index))) - t0);
            const auto x = warped_x + (velocity_x > 0 ? maximum_delta_x : 0.0);
            const auto y = warped_y + (velocity_y > 0 ? maximum_delta_y : 0.0);
            auto xi = std::floor(x);
            auto yi = std::floor(y);
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
    {"accumulate", accumulate, METH_VARARGS, nullptr},
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
