#include "pytb.cuh"



void Pytb::pytb_init()
{
    Py_Initialize();
    import_array();

    if (!Py_IsInitialized())
    {
        printf("Failed to initialize Python");
    }

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./')");

    PyObject* pModule = PyImport_ImportModule("get_data");
    if (!pModule)
    {
        printf("Unable to import Python module");
        PyErr_Print();
    }

    PyObject* pFunc = PyObject_GetAttrString(pModule, "get_data");
    if (!pFunc || !PyCallable_Check(pFunc))
    {
        printf("Unable to obtain function or function is not callable");
        PyErr_Print();
    }

    PyObject* pResult = PyObject_CallObject(pFunc, NULL);
    if (!pResult)
    {
        printf("Call to Python function failed");
        PyErr_Print();
        Py_DECREF(pFunc);
        Py_DECREF(pModule);
        Py_Finalize();
    }


    PyObject* pArray_NN_orbit = PyDict_GetItemString(pResult, "NN_orbit");
    PyObject* pArray_NL_orbit = PyDict_GetItemString(pResult, "NL_orbit");
    PyObject* pArray_Hr_hopping_val = PyDict_GetItemString(pResult, "Hr_hopping_val");
    PyObject* pArray_Hr_onsite_val = PyDict_GetItemString(pResult, "Hr_onsite_val");
    PyObject* pArray_xx = PyDict_GetItemString(pResult, "xx");

    PyObject* pArray_orbit_size = PyDict_GetItemString(pResult, "orbit_size");
    PyObject* pArray_atoms_type = PyDict_GetItemString(pResult, "atoms_type");

    PyObject* pArray_orbit_offset = PyDict_GetItemString(pResult, "orbit_offset");
    PyObject* pArray_orbit_offset2 = PyDict_GetItemString(pResult, "orbit_offset2");

    PyArrayObject* npArray_NN_orbit = reinterpret_cast<PyArrayObject*>(pArray_NN_orbit);
    PyArrayObject* npArray_NL_orbit = reinterpret_cast<PyArrayObject*>(pArray_NL_orbit);
    PyArrayObject* npArray_Hr_hopping_val = reinterpret_cast<PyArrayObject*>(pArray_Hr_hopping_val);
    PyArrayObject* npArray_Hr_onsite_val = reinterpret_cast<PyArrayObject*>(pArray_Hr_onsite_val);
    PyArrayObject* npArray_xx = reinterpret_cast<PyArrayObject*>(pArray_xx);
    PyArrayObject* npArray_orbit_size = reinterpret_cast<PyArrayObject*>(pArray_orbit_size);
    PyArrayObject* npArray_atoms_type = reinterpret_cast<PyArrayObject*>(pArray_atoms_type);
    PyArrayObject* npArray_orbit_offset = reinterpret_cast<PyArrayObject*>(pArray_orbit_offset);
    PyArrayObject* npArray_orbit_offset2 = reinterpret_cast<PyArrayObject*>(pArray_orbit_offset2);



    int* data_NN_orbit = static_cast<int*>(PyArray_DATA(npArray_NN_orbit));
    int* data_NL_orbit = static_cast<int*>(PyArray_DATA(npArray_NL_orbit));
    int* data_orbit_size = static_cast<int*>(PyArray_DATA(npArray_orbit_size));
    int* data_atoms_type = static_cast<int*>(PyArray_DATA(npArray_atoms_type));
    int* data_orbit_offset = static_cast<int*>(PyArray_DATA(npArray_orbit_offset));
    int* data_orbit_offset2 = static_cast<int*>(PyArray_DATA(npArray_orbit_offset2));
    double* data_Hr_hopping_val = static_cast<double*>(PyArray_DATA(npArray_Hr_hopping_val));
    double* data_Hr_onsite_val = static_cast<double*>(PyArray_DATA(npArray_Hr_onsite_val));
    double* data_xx = static_cast<double*>(PyArray_DATA(npArray_xx));

    int size_NN_orbit = PyArray_SIZE(npArray_NN_orbit);
    int size_NL_orbit = PyArray_SIZE(npArray_NL_orbit);
    int size_orbit_size = PyArray_SIZE(npArray_orbit_size);
    int size_atoms_type = PyArray_SIZE(npArray_atoms_type);
    int size_orbit_offset = PyArray_SIZE(npArray_orbit_offset);
    int size_orbit_offset2 = PyArray_SIZE(npArray_orbit_offset2);
    int size_Hr_hopping_val = PyArray_SIZE(npArray_Hr_hopping_val);
    int size_Hr_onsite_val = PyArray_SIZE(npArray_Hr_onsite_val);
    int size_xx = PyArray_SIZE(npArray_xx);

    NN_orbit.assign(data_NN_orbit, data_NN_orbit + size_NN_orbit);
    NL_orbit.assign(data_NL_orbit, data_NL_orbit + size_NL_orbit);
    orbit_size.assign(data_orbit_size, data_orbit_size + size_orbit_size);
    atoms_type.assign(data_atoms_type, data_atoms_type + size_atoms_type);
    orbit_offset.assign(data_orbit_offset, data_orbit_offset + size_orbit_offset);
    orbit_offset2.assign(data_orbit_offset2, data_orbit_offset2 + size_orbit_offset2);
    atoms_type.assign(data_atoms_type, data_atoms_type + size_atoms_type);
    Hr_hopping_val.assign(data_Hr_hopping_val, data_Hr_hopping_val + size_Hr_hopping_val);
    Hr_onsite_val.assign(data_Hr_onsite_val, data_Hr_onsite_val + size_Hr_onsite_val);
    xx.assign(data_xx, data_xx + size_xx);



    PyObject* p_type_number = PyDict_GetItemString(pResult, "type_number");
    PyObject* p_max_energy = PyDict_GetItemString(pResult, "max_energy");
    PyObject* p_start_energy = PyDict_GetItemString(pResult, "start_energy");
    PyObject* p_end_energy = PyDict_GetItemString(pResult, "end_energy");
    PyObject* p_volume = PyDict_GetItemString(pResult, "volume");
    PyObject* p_time_step = PyDict_GetItemString(pResult, "time_step");
    PyObject* p_num_moments = PyDict_GetItemString(pResult, "num_moments");
    PyObject* p_num_energies = PyDict_GetItemString(pResult, "num_energies");
    PyObject* p_transport_direction = PyDict_GetItemString(pResult, "transport_direction");
    PyObject* p_num_of_atoms = PyDict_GetItemString(pResult, "num_of_atoms");
    PyObject* p_num_of_orbits = PyDict_GetItemString(pResult, "num_of_orbits");
    PyObject* p_num_of_steps = PyDict_GetItemString(pResult, "num_of_steps");
    

     type_number = pyint2cint_single(p_type_number);
     max_energy =  pydouble2cdouble_single(p_max_energy);
     start_energy =  pydouble2cdouble_single(p_start_energy);
     end_energy =  pydouble2cdouble_single(p_end_energy);
     volume = pydouble2cdouble_single(p_volume);
     time_step = pydouble2cdouble_single(p_time_step);
     num_moments = pyint2cint_single(p_num_moments);
     num_energies =  pyint2cint_single(p_num_energies);
     transport_direction = pyint2cint_single(p_transport_direction);
     num_of_atoms= pyint2cint_single(p_num_of_atoms);
     num_of_orbits = pyint2cint_single(p_num_of_orbits);
     num_of_steps = pyint2cint_single(p_num_of_steps);


     dev_NN_orbit.resize(NN_orbit.size());
     dev_NL_orbit.resize(NL_orbit.size());
     dev_atoms_type.resize(atoms_type.size());
     dev_Hr_onsite_val.resize(Hr_onsite_val.size());
     dev_Hr_hopping_val.resize(Hr_hopping_val.size());
     dev_Hi_hopping_val.resize(dev_Hr_hopping_val.size());
     dev_xx.resize(xx.size());
     dev_orbit_size.resize(orbit_size.size());
     dev_orbit_offset.resize(orbit_offset.size());
     dev_orbit_offset2.resize(orbit_offset2.size());

     dev_Hr_hopping_val.resize(Hr_hopping_val.size());
     dev_Hi_hopping_val.resize(dev_Hr_hopping_val.size());
     
     dev_NN_orbit.copy_from_host(NN_orbit.data());
     dev_NL_orbit.copy_from_host(NL_orbit.data());
     dev_atoms_type.copy_from_host(atoms_type.data());
     dev_Hr_onsite_val.copy_from_host(Hr_onsite_val.data());
     dev_Hr_hopping_val.copy_from_host(Hr_hopping_val.data());
     dev_xx.copy_from_host(xx.data());
     dev_orbit_size.copy_from_host(orbit_size.data());
     dev_orbit_offset.copy_from_host(orbit_offset.data());
     dev_orbit_offset2.copy_from_host(orbit_offset2.data());


     orbit_atom_id.resize(num_of_orbits);



     for (int i = 0; i < num_of_atoms; ++i) {
         int type_id= atoms_type[i];
         int length = orbit_size[type_id];
         

         for (int a = 0; a < length;++a) {
             int offset_id = a + orbit_offset[i];
             orbit_atom_id[offset_id] = i;
         }
     }

     dev_orbit_atom_id.resize(orbit_atom_id.size());
     dev_orbit_atom_id.copy_from_host(orbit_atom_id.data());

    Py_DECREF(pArray_NN_orbit);
    Py_DECREF(pArray_NL_orbit);
    Py_DECREF(pArray_Hr_hopping_val);
    Py_DECREF(pArray_Hr_onsite_val);
    Py_DECREF(pArray_atoms_type);
    Py_DECREF(pArray_xx);
    Py_DECREF(pArray_orbit_offset);
    Py_DECREF(pArray_orbit_offset2);
    Py_DECREF(npArray_NN_orbit);
    Py_DECREF(npArray_NL_orbit);
    Py_DECREF(npArray_Hr_hopping_val);
    Py_DECREF(npArray_Hr_onsite_val);
    Py_DECREF(npArray_xx);
    Py_DECREF(npArray_orbit_size);
    Py_DECREF(npArray_atoms_type);
    Py_DECREF(npArray_orbit_offset);
    Py_DECREF(npArray_orbit_offset2);
    Py_DECREF(p_type_number);
    Py_DECREF(p_max_energy);
    Py_DECREF(p_start_energy);
    Py_DECREF(p_end_energy);
    Py_DECREF(p_num_moments);
    Py_DECREF(p_num_energies);
    Py_DECREF(p_transport_direction);
    Py_DECREF(p_volume);
    Py_DECREF(p_time_step);
    p_num_of_atoms = NULL;
    p_num_of_orbits = NULL;
    p_num_of_steps = NULL; //没法释放掉，还是指向NULL吧，反正后面不用
    Py_DECREF(pResult);
    Py_DECREF(pFunc);
    Py_DECREF(pModule);
 
    Py_Finalize();

}