#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/experimental_dtype_api.h"

#include "units_dtype.h"

Units_dtypeObject *instantiate_Units(PyObject *units)
{
    Units_dtypeObject *new = (Units_dtypeObject *)PyArrayDescr_Type.tp_new(
        (PyTypeObject *)&Units_dtype, NULL, NULL);
    if (new == NULL)
        return NULL;
    Py_IncRef(units);
    new->units = units;
    new->base.elsize = sizeof(double);
    new->base.alignment = _Alignof(double);
    return new;
}

static Units_dtypeObject *
common_instance(Units_dtypeObject *dtype1, Units_dtypeObject *dtype2)
{
    return dtype1;
}

static PyArray_DTypeMeta *
common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    if (other->type_num >= 0 && PyTypeNum_ISNUMBER(other->type_num) && !PyTypeNum_ISCOMPLEX(other->type_num) && other != &PyArray_LongDoubleDType)
    {
        Py_INCREF(cls);
        return cls;
    }
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}

static PyArray_Descr *
discover_descriptor_from_pyobject(PyArray_DTypeMeta *NPY_UNUSED(cls), PyObject *obj)
{
    PyObject *units = PyUnicode_FromString("m");
    if (units == NULL)
        return NULL;
    return (PyArray_Descr *)instantiate_Units(units);
}

static int
setitem(Units_dtypeObject *descr, PyObject *obj, char *dataptr)
{
    double value = PyFloat_AsDouble(PyNumber_Float(obj));
    memcpy(dataptr, &value, sizeof(double));
}

static PyObject *
getitem(Units_dtypeObject *descr, char *dataptr)
{
    double value;
    memcpy(&value, dataptr, sizeof(double));
    PyObject *value_obj = PyFloat_FromDouble(value);
    if (value_obj == NULL)
        return NULL;
    return value_obj;
}

static PyType_Slot Units_dtype_Slots[] = {
    {NPY_DT_common_instance, &common_instance},
    {NPY_DT_common_dtype, &common_dtype},
    {NPY_DT_discover_descr_from_pyobject, &discover_descriptor_from_pyobject},
    {NPY_DT_setitem, &setitem},
    {NPY_DT_getitem, &getitem},
    {0, NULL}};

static PyObject *
Units_dtype_new(PyTypeObject *NPY_UNUSED(cls), PyObject *args, PyObject *kwds)
{
}