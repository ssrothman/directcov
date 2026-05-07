#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <directcov/DirectCov.h>

namespace py = pybind11;

py::buffer_info get_buffer(directcov::DirectCov& dc) {
    return py::buffer_info(
        dc.data_(),                       // Pointer to buffer
        sizeof(double),                  // Size of one scalar
        py::format_descriptor<double>::format(), // Python struct-style format descriptor
        dc.get_buffer_shape_().size(), // Number of dimensions
        dc.get_buffer_shape_(),                      // Shape of the buffer
        dc.get_buffer_strides_(),                     // Strides of the buffer
        true
    );
}

PYBIND11_MODULE(_core, m) {
    py::class_<directcov::DirectCov>(m, "DirectCov", py::buffer_protocol())
        .def_buffer([](directcov::DirectCov& dc) -> py::buffer_info {
            if(!dc.isFinalized()){
                throw std::runtime_error("DirectCov must be finalized before buffer access");
            }
            return get_buffer(dc);
        })
        .def("view", [](directcov::DirectCov& dc){
            if(!dc.isFinalized()){
                throw std::runtime_error("DirectCov is not finalized. Call finalize() before accessing the buffer.");
            }
            return get_buffer(dc);
        })
        .def(py::init<const std::vector<size_t>&, size_t>(), "Initialize DirectCov with shape",
             py::arg("shape"), py::arg("product_batch"))
        .def("fillEvents", &directcov::DirectCov::fillEvents<double, int8_t, int8_t>,
                "Add multiple events to the covariance matrix using NumPy arrays",
                py::arg("indices").noconvert(),
                py::arg("values").noconvert(), 
                py::arg("event_ids").noconvert())
        .def("fillEvents", &directcov::DirectCov::fillEvents<double, int16_t, int16_t>,
                "Add multiple events to the covariance matrix using NumPy arrays",
                py::arg("indices").noconvert(),
                py::arg("values").noconvert(), 
                py::arg("event_ids").noconvert())
        .def("fillEvents", &directcov::DirectCov::fillEvents<double, int32_t, int32_t>,
                "Add multiple events to the covariance matrix using NumPy arrays",
                py::arg("indices").noconvert(),
                py::arg("values").noconvert(), 
                py::arg("event_ids").noconvert())
        .def("fillEvents", &directcov::DirectCov::fillEvents<double, int32_t, int64_t>,
                "Add multiple events to the covariance matrix using NumPy arrays",
                py::arg("indices").noconvert(),
                py::arg("values").noconvert(), 
                py::arg("event_ids").noconvert())
        .def("fillEvents", &directcov::DirectCov::fillEvents<double, int64_t, int32_t>,
                "Add multiple events to the covariance matrix using NumPy arrays",
                py::arg("indices").noconvert(),
                py::arg("values").noconvert(), 
                py::arg("event_ids").noconvert())
        .def("fillEvents", &directcov::DirectCov::fillEvents<double, int64_t, int64_t>,
                "Add multiple events to the covariance matrix using NumPy arrays",
                py::arg("indices").noconvert(),
                py::arg("values").noconvert(), 
                py::arg("event_ids").noconvert())
        .def("fillEvents", &directcov::DirectCov::fillEvents<double, uint32_t, int32_t>,
                "Add multiple events to the covariance matrix using NumPy arrays",
                py::arg("indices").noconvert(),
                py::arg("values").noconvert(), 
                py::arg("event_ids").noconvert())
        .def("fillEvents", &directcov::DirectCov::fillEvents<double, int32_t, uint32_t>,
                "Add multiple events to the covariance matrix using NumPy arrays",
                py::arg("indices").noconvert(),
                py::arg("values").noconvert(), 
                py::arg("event_ids").noconvert())
        .def("fillEvents", &directcov::DirectCov::fillEvents<double, uint32_t, uint32_t>,
                "Add multiple events to the covariance matrix using NumPy arrays",
                py::arg("indices").noconvert(),
                py::arg("values").noconvert(), 
                py::arg("event_ids").noconvert())
        .def("fillEvents", &directcov::DirectCov::fillEvents<double, int32_t, uint64_t>,
                "Add multiple events to the covariance matrix using NumPy arrays",
                py::arg("indices").noconvert(),
                py::arg("values").noconvert(), 
                py::arg("event_ids").noconvert())
        .def("fillEvents", &directcov::DirectCov::fillEvents<double, uint32_t, int64_t>,
                "Add multiple events to the covariance matrix using NumPy arrays",
                py::arg("indices").noconvert(),
                py::arg("values").noconvert(), 
                py::arg("event_ids").noconvert())
        .def("fillEvents", &directcov::DirectCov::fillEvents<double, uint32_t, uint64_t>,
                "Add multiple events to the covariance matrix using NumPy arrays",
                py::arg("indices").noconvert(),
                py::arg("values").noconvert(), 
                py::arg("event_ids").noconvert())
        .def("fillEvents", &directcov::DirectCov::fillEvents<double, uint64_t, int32_t>,
                "Add multiple events to the covariance matrix using NumPy arrays",
                py::arg("indices").noconvert(),
                py::arg("values").noconvert(), 
                py::arg("event_ids").noconvert())
        .def("fillEvents", &directcov::DirectCov::fillEvents<double, int64_t, uint32_t>,
                "Add multiple events to the covariance matrix using NumPy arrays",
                py::arg("indices").noconvert(),
                py::arg("values").noconvert(), 
                py::arg("event_ids").noconvert())
        .def("fillEvents", &directcov::DirectCov::fillEvents<double, uint64_t, uint32_t>,
                "Add multiple events to the covariance matrix using NumPy arrays",
                py::arg("indices").noconvert(),
                py::arg("values").noconvert(), 
                py::arg("event_ids").noconvert())
        .def("fillEvents", &directcov::DirectCov::fillEvents<double, uint64_t, int64_t>,
                "Add multiple events to the covariance matrix using NumPy arrays",
                py::arg("indices").noconvert(),
                py::arg("values").noconvert(), 
                py::arg("event_ids").noconvert())
        .def("fillEvents", &directcov::DirectCov::fillEvents<double, int64_t, uint64_t>,
                "Add multiple events to the covariance matrix using NumPy arrays",
                py::arg("indices").noconvert(),
                py::arg("values").noconvert(), 
                py::arg("event_ids").noconvert())
        .def("fillEvents", &directcov::DirectCov::fillEvents<double, uint64_t, uint64_t>,
                "Add multiple events to the covariance matrix using NumPy arrays",
                py::arg("indices").noconvert(),
                py::arg("values").noconvert(), 
                py::arg("event_ids").noconvert())
        .def("fillEvents", &directcov::DirectCov::fillEvents<float, int8_t, int8_t>,
                "Add multiple events to the covariance matrix using NumPy arrays",
                py::arg("indices").noconvert(),
                py::arg("values").noconvert(), 
                py::arg("event_ids").noconvert())
        .def("fillEvents", &directcov::DirectCov::fillEvents<float, int16_t, int16_t>,
                "Add multiple events to the covariance matrix using NumPy arrays",
                py::arg("indices").noconvert(),
                py::arg("values").noconvert(), 
                py::arg("event_ids").noconvert())
        .def("fillEvents", &directcov::DirectCov::fillEvents<float, int32_t, int32_t>,
                "Add multiple events to the covariance matrix using NumPy arrays",
                py::arg("indices").noconvert(),
                py::arg("values").noconvert(), 
                py::arg("event_ids").noconvert())
        .def("fillEvents", &directcov::DirectCov::fillEvents<float, int64_t, int32_t>,
                "Add multiple events to the covariance matrix using NumPy arrays",
                py::arg("indices").noconvert(),
                py::arg("values").noconvert(), 
                py::arg("event_ids").noconvert())
        .def("fillEvents", &directcov::DirectCov::fillEvents<float, int32_t, int64_t>,
                "Add multiple events to the covariance matrix using NumPy arrays",
                py::arg("indices").noconvert(),
                py::arg("values").noconvert(), 
                py::arg("event_ids").noconvert())
        .def("fillEvents", &directcov::DirectCov::fillEvents<float, int64_t, int64_t>,
                "Add multiple events to the covariance matrix using NumPy arrays",
                py::arg("indices").noconvert(),
                py::arg("values").noconvert(), 
                py::arg("event_ids").noconvert())
        .def("finalize", &directcov::DirectCov::finalize,
                "Finalize the covariance matrix, making it ready for use")
        .def("clear", &directcov::DirectCov::clear,
                "Clear the covariance matrix")
        .def("shape", &directcov::DirectCov::shape,
                "Get the shape of the covariance matrix")
        .def("strides", &directcov::DirectCov::strides,
                "Get the strides of the covariance matrix")
        .def("size", &directcov::DirectCov::size,
                "Get the total number of elements in the covariance matrix")
        .def("isFinalized", &directcov::DirectCov::isFinalized,
                "Check if the covariance matrix has been finalized");
}
