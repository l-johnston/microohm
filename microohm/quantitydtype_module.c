#define Py_SSIZE_T_CLEAN
#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL QuantityDType_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"
#include "quantitydtype.h"
#include "umath.h"

static struct PyModuleDef quantitydtype = {
    PyModuleDef_HEAD_INIT,
    .m_name = "quantitydtype",
    .m_size = -1};

PyMODINIT_FUNC
PyInit_quantitydtype(void)
{
    if (_import_array() < 0)
        return NULL;
    if (import_experimental_dtype_api(5) < 0)
        return NULL;
    PyObject *m = PyModule_Create(&quantitydtype);
    if (m == NULL)
        return NULL;
    PyObject *quantity_module = PyImport_ImportModule("microohm.quantity");
    if (quantity_module == NULL)
        goto error;
    Quantity_Type = PyObject_GetAttrString(quantity_module, "Quantity");
    Py_DECREF(quantity_module);
    if (Quantity_Type == NULL)
        goto error;
    if (init_QuantityDType() < 0)
        goto error;
    if (PyModule_AddObject(m, "QuantityDType", (PyObject *)&QuantityDType) < 0)
        goto error;
    if (init_multiply_ufunc() < 0)
        goto error;
    if (init_divide_ufunc() < 0)
        goto error;
    if (init_add_ufunc() < 0)
        goto error;
    if (init_subtract_ufunc() < 0)
        goto error;
    if (init_negative_ufunc() < 0)
        goto error;
    if (init_absolute_ufunc() < 0)
        goto error;
    if (init_greater_ufunc() < 0)
        goto error;
    if (init_less_ufunc() < 0)
        goto error;
    if (init_maximum_ufunc() < 0)
        goto error;
    if (init_minimum_ufunc() < 0)
        goto error;
    return m;
error:
    Py_DECREF(m);
    return NULL;
}
