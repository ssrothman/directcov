#ifndef DIRECTCOV_DIRECTCOV_H
#define DIRECTCOV_DIRECTCOV_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace directcov {

class DirectCov {
public:
    DirectCov(const std::vector<size_t>& shape) {
        //setup shape and strides
        shape_ = shape;
        N_ = 1;
        for (size_t dim : shape_) {
            N_ *= dim;
        }
        strides_.resize(shape_.size());
        strides_[shape_.size() - 1] = 1;
        for (size_t i = shape_.size() - 1; i > 0; --i) {
            strides_[i - 1] = strides_[i] * shape_[i];
        }

        //setup buffer shape and strides
        buffer_shape_.resize(shape_.size()*2);
        for (size_t i = 0; i < shape_.size(); ++i) {
            buffer_shape_[i] = shape_[i];
        }
        for (size_t i = 0; i < shape_.size(); ++i) {
            buffer_shape_[shape_.size() + i] = shape_[i];
        }
        buffer_strides_.resize(shape_.size()*2);
        buffer_strides_[buffer_shape_.size() - 1] = 1;
        for (size_t i=buffer_shape_.size() - 1; i > 0; --i) {
            buffer_strides_[i - 1] = buffer_strides_[i] * buffer_shape_[i];
        }
        for(size_t i=0; i<buffer_strides_.size(); ++i){
            buffer_strides_[i] *= sizeof(double);
        }

        //reshape internal matrices 
        cov_.resize(N_, N_);
        current_event_vec_.resize(N_);

        //initialize
        this->clear();
    }

    DirectCov(const DirectCov& other) = default;
    DirectCov& operator=(const DirectCov& other) = default;
    DirectCov(DirectCov&& other) = default;
    DirectCov& operator=(DirectCov&& other) = default;

    const double& at_(size_t i, size_t j) const {
        return cov_(i, j);
    }

    const double& at(const std::vector<size_t>& indices_i, const std::vector<size_t>& indices_j) const {
        size_t idx_i = 0;
        size_t idx_j = 0;
        for (size_t d = 0; d < shape_.size(); ++d) {
            idx_i += indices_i[d] * strides_[d];
            idx_j += indices_j[d] * strides_[d];
        }
        return this->at_(idx_i, idx_j);
    }

    void add_(size_t i, size_t j, double value) {
        cov_(i, j) += value;
    }

    void add(const std::vector<size_t>& indices_i, const std::vector<size_t>& indices_j, double value) {
        size_t idx_i = 0;
        size_t idx_j = 0;
        for (size_t d = 0; d < shape_.size(); ++d) {
            idx_i += indices_i[d] * strides_[d];
            idx_j += indices_j[d] * strides_[d];
        }
        this->add_(idx_i, idx_j, value);
    }

    void addEvent_(const Eigen::VectorXd& vec){
        cov_ += vec * vec.transpose();
    }

    void addEvent(const std::vector<std::vector<size_t>>& indices, 
                  const std::vector<double>& values) {

        Eigen::VectorXd vec(N_);
        vec.setZero();

        for (size_t i = 0; i < indices.size(); ++i) {
            size_t idx = 0;
            for (size_t d = 0; d < shape_.size(); ++d) {
                idx += indices[i][d] * strides_[d];
            }
            vec[idx] += values[i];
        }
        this->addEvent_(vec);
    }

    void fillEvents(const std::vector<std::vector<size_t>>& indices,
                       const std::vector<double>& values,
                       const std::vector<size_t>& event_ids){
        /*
         * Each event can consist of many columns
         */
        for (size_t i = 0; i < indices.size(); ++i) {
            if (first_event_ || current_event_id_ != event_ids[i]) {
                if (!first_event_) {
                    this->addEvent_(current_event_vec_);
                } else {
                    first_event_ = false;
                }
                current_event_id_ = event_ids[i];
                new_event();
            }

            accumulate_event(indices[i], values[i]);
        }
    }
    
    template <typename FLOAT, typename INT>
    void fillEvents(const py::array_t<INT>& indices_buffer,
                         const py::array_t<FLOAT>& values_buffer,
                         const py::array_t<INT>& event_ids_buffer) {
        py::buffer_info indices_info = indices_buffer.request();
        py::buffer_info values_info = values_buffer.request();
        py::buffer_info event_ids_info = event_ids_buffer.request();

        if (indices_info.ndim != 2){
            throw std::runtime_error("Indices buffer must be 2D."); 
        }
        if (values_info.ndim != 1){
            throw std::runtime_error("Values buffer must be 1D.");
        }
        if (event_ids_info.ndim != 1){
            throw std::runtime_error("Event IDs buffer must be 1D.");
        }
        if (indices_info.shape[0] != values_info.shape[0] || 
            indices_info.shape[0] != event_ids_info.shape[0]) {
            throw std::runtime_error("All buffers must have the same length.");
        }
        if (indices_info.shape[1] != shape_.size()) {
            throw std::runtime_error("Indices buffer must have the same number of columns as the shape of the covariance matrix.");
        }

        auto indices = indices_buffer.template unchecked<2>();
        auto values = values_buffer.template unchecked<1>();
        auto event_ids = event_ids_buffer.template unchecked<1>();

        for (py::ssize_t i = 0; i < indices_info.shape[0]; ++i) {
            if (first_event_ || current_event_id_ != event_ids(i)) {
                if (!first_event_) {
                    this->addEvent_(current_event_vec_);
                } else {
                    first_event_ = false;
                }
                current_event_id_ = event_ids(i);
                new_event();
            }
            
            py::ssize_t idx = 0;
            for (py::ssize_t d = 0; d < shape_.size(); ++d) {
                idx += indices(i, d) * strides_[d];
            }
            current_event_vec_[idx] += values(i);
        }
    }

    void finalize() {
        this->addEvent_(current_event_vec_);
        current_event_id_ = 0;
        first_event_ = true;
        new_event();
    }

    void clear() {
        cov_.setZero();
        current_event_id_ = 0;
        first_event_ = true;
        current_event_vec_.setZero();
    }

    const std::vector<size_t>& shape() const {
        return shape_;
    }

    const std::vector<size_t>& strides() const {
        return strides_;
    }

    size_t size() const {
        return N_;
    }

    double* data_() {
        return cov_.data();
    }

    const std::vector<size_t>& get_buffer_shape_() const {
        return buffer_shape_;
    }

    const std::vector<size_t>& get_buffer_strides_() const {
        return buffer_strides_;
    }

    bool isFinalized() const {
        return !first_event_;
    }

private:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t N_;
    Eigen::MatrixXd cov_;

    size_t current_event_id_;
    bool first_event_;
    Eigen::VectorXd current_event_vec_;

    std::vector<size_t> buffer_shape_;
    std::vector<size_t> buffer_strides_;

    void new_event(){
        current_event_vec_.setZero();
    }

    void accumulate_event(const std::vector<size_t>& indices, 
                          double value) {
        size_t idx = 0;
        for (size_t d = 0; d < shape_.size(); ++d) {
            idx += indices[d] * strides_[d];
        }
        current_event_vec_[idx] += value;
    }
};

};

#endif
