#pragma once

#include <iostream>
#include <Python.h>
#include <vector>
#include "gpu_vector.cuh"
#include "numpy/arrayobject.h"

class Pytb {
public:
    void pytb_init();

    std::vector<int> NN_orbit;
    std::vector<int> NL_orbit;
    
    std::vector<int> orbit_size;
    std::vector<int> atoms_type;
    std::vector<double> Hr_onsite_val;
    std::vector<double> Hr_hopping_val;
    std::vector<double> xx;
    std::vector<int> orbit_offset;
    std::vector<int> orbit_offset2;
    std::vector<int> orbit_atom_id;


   // GPU_Vector<int> dev_NN;
    //GPU_Vector<int> dev_NL;
    GPU_Vector<int> dev_NN_orbit;
    GPU_Vector<int> dev_NL_orbit;
    GPU_Vector<int> dev_orbit_size;
    GPU_Vector<int> dev_atoms_type;
    GPU_Vector<double> dev_Hr_onsite_val;
    GPU_Vector<double> dev_Hr_hopping_val;
    GPU_Vector<double> dev_Hi_hopping_val;
    GPU_Vector<double> dev_xx;
    GPU_Vector<int> dev_orbit_offset;
    GPU_Vector<int> dev_orbit_offset2;
    GPU_Vector<int> dev_orbit_atom_id;

    double max_energy;
    double start_energy;
    double end_energy;
    double volume;
    double time_step;
    int num_moments;
    int num_energies;
    int transport_direction;
    int num_of_atoms;
    int num_of_orbits;
    int num_of_steps;
    int type_number;
    

    std::vector<int> pyint2cint(PyObject* pyList) {
        std::vector<int> cppArray;

        if (!PyList_Check(pyList)) {
            PyErr_SetString(PyExc_TypeError, "Expected a list");
            return cppArray;
        }
        Py_ssize_t size = PyList_Size(pyList);
        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject* item = PyList_GetItem(pyList, i);
            if (!PyLong_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "List contains non-integer values");
                return cppArray;
            }
            int value = PyLong_AsLong(item);
            cppArray.push_back(value);
        }

        return cppArray;
    }

    std::vector<double> pydouble2cdouble(PyObject* pyList) {
        std::vector<double> cppArray;
        if (!PyList_Check(pyList)) {
            PyErr_SetString(PyExc_TypeError, "Expected a list");
            return cppArray;
        }
        Py_ssize_t size = PyList_Size(pyList);
        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject* item = PyList_GetItem(pyList, i);
            if (!PyFloat_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "List contains non-float values");
                return cppArray;
            }
            double value = PyFloat_AsDouble(item);
            cppArray.push_back(value);
        }

        return cppArray;
    }


    int pyint2cint_single(PyObject* pResult) {
        if (PyLong_Check(pResult)) {
            long intValue = PyLong_AsLong(pResult);
            return intValue;
        }
        else {
            PyErr_SetString(PyExc_TypeError, "Expected an integer");
            PyErr_Print();
            return 0;
        }
    }

    double pydouble2cdouble_single(PyObject* pResult) {
        if (PyFloat_Check(pResult)) {
            double floatValue = PyFloat_AsDouble(pResult);
            return floatValue;
        }
        else {
            PyErr_SetString(PyExc_TypeError, "Expected a float");
            PyErr_Print();
            return 0.0;
        }

    }
};



