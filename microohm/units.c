#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL unitdtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"

static struct PyModuleDef unitsmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "units",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit_units(void)
{
    if (_import_array() < 0)
        return NULL;
    if (import_experimental_dtype_api(5) < 0)
        return NULL;
    PyObject *m = PyModule_Create(&unitsmodule);
    if (m == NULL)
        return NULL;
    return m;
}
