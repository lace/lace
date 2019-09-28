#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <boost/cstdint.hpp>
#include <boost/array.hpp>
using boost::array;
using boost::uint32_t;

#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/array.hpp>
#include <vector>
#include <map>
#include <algorithm>
#include <boost/algorithm/string.hpp>

class LoadObjException : public std::exception
{
public:
    LoadObjException(std::string m = "loadObjException!") : msg(m) {}
    ~LoadObjException() throw() {}
    const char *what() const throw() { return msg.c_str(); }

private:
    std::string msg;
};

static PyObject *
objutils_read(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *LoadObjError;

static PyMethodDef objutils_methods[] = {
    {"read", (PyCFunction)objutils_read, METH_VARARGS | METH_KEYWORDS, "Read an OBJ file."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "objutils",       /* m_name */
    NULL,             /* m_doc */
    -1,               /* m_size */
    objutils_methods, /* m_methods */
    NULL,             /* m_reload */
    NULL,             /* m_traverse */
    NULL,             /* m_clear */
    NULL,             /* m_free */
};
#endif

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_objutils(void)
#else
initobjutils(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *m = PyModule_Create(&moduledef);
#else
    PyObject *m = Py_InitModule("objutils", objutils_methods);
#endif
    if (m == NULL)
    {
#if PY_MAJOR_VERSION >= 3
        return NULL;
#else
        return;
#endif
    }

    import_array();
    LoadObjError = PyErr_NewException(const_cast<char *>("objutils.LoadObjError"), NULL, NULL);
    Py_INCREF(LoadObjError);
    PyModule_AddObject(m, "LoadObjError", LoadObjError);
#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}

static PyObject *
objutils_read(PyObject *self, PyObject *args, PyObject *keywds)
{
    try
    {
        char py_objpatharr[256];
        char *py_objpath = static_cast<char *>(py_objpatharr);

        static char *kwlist[] = {"obj_path", NULL};

        if (!PyArg_ParseTupleAndKeywords(args, keywds, "s", kwlist, &py_objpath))
            return NULL;

        std::ifstream obj_is(py_objpath, std::ios_base::binary | std::ios_base::in);
        if (!obj_is)
        {
            PyErr_SetString(PyExc_ValueError, "Could not load file");
            return NULL;
        }

        std::vector<double> v;
        std::vector<double> vt;
        std::vector<double> vn;
        std::vector<double> vc;
        std::vector<uint32_t> f;
        std::vector<uint32_t> ft;
        std::vector<uint32_t> fn;
        v.reserve(30000);
        vt.reserve(30000);
        vn.reserve(30000);
        vc.reserve(30000);
        f.reserve(100000);
        ft.reserve(100000);
        fn.reserve(100000);
        std::map<std::string, std::vector<uint32_t>> segm;

        bool next_v_is_land = false;
        bool has_vertex_colors = true;
        std::string land_name("");
        std::map<std::string, uint32_t> landm;

        std::string line;
        std::vector<std::string> curr_segm;
        std::string mtl_path("");
        uint32_t len_vt = 3;
        uint32_t len_vc = 3;

        while (getline(obj_is, line))
        {
            boost::erase_all(line, "\r"); // argh windows line endings
            if (line.substr(0, 7) == "mtllib ")
            {
                mtl_path = line.substr(6);
            }
            else if (line.substr(0, 2) == "g ")
            {
                curr_segm.clear();
                std::istringstream is(line.substr(2));
                std::string segm_name;
                while (is >> segm_name)
                {
                    if (segm.find(segm_name) == segm.end())
                        segm[segm_name] = std::vector<uint32_t>();
                    curr_segm.push_back(segm_name);
                }
            }
            else if (line.substr(0, 3) == "vt ")
            {
                std::istringstream is(line.substr(2));
                uint32_t orig_vt_len = (uint32_t)vt.size();
                for (int i = 0; i < 2; ++i)
                {
                    double x;
                    if (!(is >> x))
                    {
                        throw LoadObjException("Malformed OBJ file: could not parse vertex uvs");
                    }
                    vt.push_back(x);
                }
                len_vt = (uint32_t)vt.size() - orig_vt_len;
            }
            else if (line.substr(0, 3) == "vn ")
            {
                std::istringstream is(line.substr(2));
                for (int i = 0; i < 3; ++i)
                {
                    double x;
                    if (!(is >> x))
                    {
                        throw LoadObjException("Malformed OBJ file: could not parse vertex normals");
                    }
                    vn.push_back(x);
                }
            }
            else if (line.substr(0, 2) == "f ")
            {
                std::istringstream is(line.substr(1));
                std::istream_iterator<std::string> it(is);
                std::vector<uint32_t> localf, localfn, localft;
                for (; it != std::istream_iterator<std::string>(); ++it)
                {
                    // valid:  v   v/vt   v/vt/vn   v//vn
                    uint32_t counter = 0;
                    std::istringstream unparsed_face(*it);
                    std::string el;
                    size_t n_slashes = std::count(it->begin(), it->end(), '/');
                    while (std::getline(unparsed_face, el, '/'))
                    {
                        if (el.size() > 0)
                        { // if the element has contents
                            const char *elptr = el.c_str();
                            char *endptr;
                            long value = strtol(elptr, &endptr, 10); /* 10 is the base */
                            if (elptr == endptr)
                            {
                                throw LoadObjException("Malformed OBJ file: could not parse face");
                            }
                            if (counter == 0)
                            {
                                localf.push_back(value);
                            }
                            else if (counter == 1)
                            {
                                localft.push_back(value);
                            }
                            else if (counter == 2)
                            {
                                localfn.push_back(value);
                            }
                        }
                        counter++;
                    }
                    if (n_slashes == 1 && localft.size() == 0)
                    {
                        throw LoadObjException("Malformed OBJ file: could not parse face (v/)");
                    }
                    if (n_slashes == 2 && localfn.size() == 0)
                    {
                        throw LoadObjException("Malformed OBJ file: could not parse face (v//)");
                    }
                }
                if (localf.size() > 0)
                {
                    for (int i = 1; i < (localf.size() - 1); ++i)
                    {
                        f.push_back(localf[0] - 1);
                        f.push_back(localf[i] - 1);
                        f.push_back(localf[i + 1] - 1);
                        uint32_t face_index = ((uint32_t)f.size() / 3) - 1;
                        for (std::vector<std::string>::iterator it = curr_segm.begin(); it != curr_segm.end(); ++it)
                            segm.find(*it)->second.push_back(face_index);
                    }
                }
                if (localft.size() > 0)
                {
                    if (localft.size() != localf.size())
                    {
                        throw LoadObjException("Malformed OBJ file: could not parse face (len(ft) != len(f))");
                    }
                    for (int i = 1; i < (localft.size() - 1); ++i)
                    {
                        ft.push_back(localft[0] - 1);
                        ft.push_back(localft[i] - 1);
                        ft.push_back(localft[i + 1] - 1);
                    }
                }
                if (localfn.size() > 0)
                {
                    if (localfn.size() != localf.size())
                    {
                        throw LoadObjException("Malformed OBJ file: could not parse face (len(fn) != len(f))");
                    }
                    for (int i = 1; i < (localfn.size() - 1); ++i)
                    {
                        fn.push_back(localfn[0] - 1);
                        fn.push_back(localfn[i] - 1);
                        fn.push_back(localfn[i + 1] - 1);
                    }
                }
            }
            else if (line.substr(0, 2) == "v ")
            {
                std::istringstream is(line.substr(1));
                for (int i = 0; i < 3; ++i)
                {
                    double x;
                    if (!(is >> x))
                    {
                        throw LoadObjException("Malformed OBJ file: could not parse vertices");
                    }
                    v.push_back(x);
                }

                if (has_vertex_colors)
                {
                    for (int i = 0; i < len_vc; ++i)
                    {
                        double c;
                        if (!(is >> c))
                        {
                            // If there is no value after the vertex values, it simply has
                            // no vertex color data, sets a flag, and continues.
                            // Otherwise, it is an error
                            if (i > 0)
                            {
                                throw LoadObjException("Malformed OBJ file: could not parse vertex colors");
                            }
                            else
                            {
                                has_vertex_colors = false;
                                break;
                            }
                        }
                        vc.push_back(c);
                    }
                }

                if (next_v_is_land)
                {
                    next_v_is_land = false;
                    landm[land_name.c_str()] = (uint32_t)v.size() / 3 - 1;
                }
            }
            else if (line.substr(0, 9) == "#landmark")
            {
                next_v_is_land = true;
                land_name = line.substr(10);
            }
            else if (line.substr(0, 3) == "vp " ||
                     line.substr(0, 7) == "usemtl " ||
                     line.substr(0, 2) == "o " ||
                     line.substr(0, 1) == "s")
            { // spec allows `s INT` or `s off`; 3dMD puts in a line with just `s`
                // allowed, but unused
            }
            else if (line.substr(0, 1) == "#" || line.find_first_not_of(" \t\v\f") == std::string::npos)
            {
                // comment or blank line
            }
            else
            {
                throw LoadObjException("Malformed OBJ file: unknown line type '" + line + "'");
            }
        }

        uint32_t n_v = (uint32_t)v.size() / 3;
        uint32_t n_vt = (uint32_t)vt.size() / len_vt;
        uint32_t n_vn = (uint32_t)vn.size() / 3;
        uint32_t n_vc = (uint32_t)vc.size() / len_vc;
        uint32_t n_f = (uint32_t)f.size() / 3;
        uint32_t n_ft = (uint32_t)ft.size() / 3;
        uint32_t n_fn = (uint32_t)fn.size() / 3;
        npy_intp v_dims[] = {n_v, 3};
        npy_intp vn_dims[] = {n_vn, 3};
        npy_intp vc_dims[] = {n_vc, len_vc};
        npy_intp vt_dims[] = {n_vt, len_vt};
        npy_intp f_dims[] = {n_f, 3};
        npy_intp ft_dims[] = {n_ft, 3};
        npy_intp fn_dims[] = {n_fn, 3};
        /*
        // XXX Memory from vectors get deallocated!
        PyObject *py_v = PyArray_SimpleNewFromData(2, v_dims, NPY_DOUBLE, v.data());
        PyObject *py_vt = PyArray_SimpleNewFromData(2, vt_dims, NPY_DOUBLE, vt.data());
        PyObject *py_vn = PyArray_SimpleNewFromData(2, vn_dims, NPY_DOUBLE, vn.data());
        PyObject *py_f = PyArray_SimpleNewFromData(2, f_dims, NPY_UINT32, f.data());
        PyObject *py_ft = PyArray_SimpleNewFromData(2, ft_dims, NPY_UINT32, ft.data());
        PyObject *py_fn = PyArray_SimpleNewFromData(2, fn_dims, NPY_UINT32, fn.data());
        */
        // The following copy would be faster in C++11 with move semantics
        PyArrayObject *py_v = (PyArrayObject *)PyArray_SimpleNew(2, v_dims, NPY_DOUBLE);
        std::copy(v.begin(), v.end(), reinterpret_cast<double *>(PyArray_DATA(py_v)));
        PyArrayObject *py_vt = (PyArrayObject *)PyArray_SimpleNew(2, vt_dims, NPY_DOUBLE);
        std::copy(vt.begin(), vt.end(), reinterpret_cast<double *>(PyArray_DATA(py_vt)));
        PyArrayObject *py_vn = (PyArrayObject *)PyArray_SimpleNew(2, vn_dims, NPY_DOUBLE);
        std::copy(vn.begin(), vn.end(), reinterpret_cast<double *>(PyArray_DATA(py_vn)));
        PyArrayObject *py_vc = (PyArrayObject *)PyArray_SimpleNew(2, vc_dims, NPY_DOUBLE);
        std::copy(vc.begin(), vc.end(), reinterpret_cast<double *>(PyArray_DATA(py_vc)));
        PyArrayObject *py_f = (PyArrayObject *)PyArray_SimpleNew(2, f_dims, NPY_UINT32);
        std::copy(f.begin(), f.end(), reinterpret_cast<uint32_t *>(PyArray_DATA(py_f)));
        PyArrayObject *py_ft = (PyArrayObject *)PyArray_SimpleNew(2, ft_dims, NPY_UINT32);
        std::copy(ft.begin(), ft.end(), reinterpret_cast<uint32_t *>(PyArray_DATA(py_ft)));
        PyArrayObject *py_fn = (PyArrayObject *)PyArray_SimpleNew(2, fn_dims, NPY_UINT32);
        std::copy(fn.begin(), fn.end(), reinterpret_cast<uint32_t *>(PyArray_DATA(py_fn)));

        PyObject *py_landm = PyDict_New();
        for (std::map<std::string, uint32_t>::iterator it = landm.begin(); it != landm.end(); ++it)
            PyDict_SetItemString(py_landm, it->first.c_str(), Py_BuildValue("l", it->second));

        PyObject *py_segm = PyDict_New();
        for (std::map<std::string, std::vector<uint32_t>>::iterator it = segm.begin(); it != segm.end(); ++it)
        {
            uint32_t n = (uint32_t)it->second.size();
            npy_intp dims[] = {n};
            PyArrayObject *temp = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_UINT32);
            std::copy(it->second.begin(), it->second.end(), reinterpret_cast<uint32_t *>(PyArray_DATA(temp)));
            PyDict_SetItemString(py_segm, it->first.c_str(), Py_BuildValue("N", temp));
        }

        return Py_BuildValue("NNNNNNNsNN", py_v, py_vt, py_vn, py_vc, py_f, py_ft, py_fn, mtl_path.c_str(), py_landm, py_segm);
    }
    catch (LoadObjException &e)
    {
        PyErr_SetString(LoadObjError, e.what());
        return NULL;
    }
}
