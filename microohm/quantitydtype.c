#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL QuantityDType_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/experimental_dtype_api.h"
#include "quantitydtype.h"
#include "casts.h"

PyObject *Quantity_Type = NULL;

static double
get_value(PyObject *scalar)
{
    PyObject *value = PyObject_GetAttrString(scalar, "real");
    if (value == NULL)
        return -1;
    double res = PyFloat_AsDouble(value);
    Py_DECREF(value);
    return res;
}

QuantityDTypeObject *
instantiate_QuantityDType(PyObject *units)
{
    QuantityDTypeObject *new = (QuantityDTypeObject *)PyArrayDescr_Type.tp_new((PyTypeObject *)&QuantityDType, NULL, NULL);
    if (new == NULL)
        return NULL;
    Py_INCREF(units);
    new->units = units;
    new->base.elsize = sizeof(double);
    new->base.alignment = _Alignof(double);
    return new;
}

static QuantityDTypeObject *
common_instance(QuantityDTypeObject *dtype1, QuantityDTypeObject *dtype2)
{
    return dtype1;
}

static PyArray_DTypeMeta *
common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    Py_INCREF(cls);
    return cls;
}

static PyArray_Descr *
discover_descriptor(PyArray_DTypeMeta *NPY_UNUSED(cls), PyObject *obj)
{
    return (PyArray_Descr *)instantiate_QuantityDType(PyUnicode_FromString(""));
}

static int
setitem(QuantityDTypeObject *descr, PyObject *obj, char *dataptr)
{
    double value = get_value(obj);
    if (value == -1 && PyErr_Occurred())
        return -1;
    memcpy(dataptr, &value, sizeof(double));
    return 0;
}

static PyObject *
getitem(QuantityDTypeObject *descr, char *dataptr)
{
    double value;
    memcpy(&value, dataptr, sizeof(double));
    PyObject *value_obj = PyFloat_FromDouble(value);
    if (value_obj == NULL)
        return NULL;
    PyObject *res = PyObject_CallFunctionObjArgs(Quantity_Type, value_obj, descr->units, NULL);
    Py_DECREF(value_obj);
    return res;
}

static QuantityDTypeObject *
ensure_canonical(QuantityDTypeObject *self)
{
    Py_INCREF(self);
    return self;
}

static PyType_Slot QuantityDType_Slots[] = {
    {NPY_DT_common_instance, &common_instance},
    {NPY_DT_common_dtype, &common_dtype},
    {NPY_DT_discover_descr_from_pyobject, &discover_descriptor},
    {NPY_DT_setitem, &setitem},
    {NPY_DT_getitem, &getitem},
    {NPY_DT_ensure_canonical, &ensure_canonical},
    {0, NULL}};

static PyObject *
QuantityDType_new(PyTypeObject *NPY_UNUSED(cls), PyObject *args, PyObject *kwargs)
{
    static char *kwds[] = {"units", NULL};
    PyObject *units = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwds, &units))
        return NULL;
    if (units == NULL)
        units = PyUnicode_FromString("1");
    return (PyObject *)instantiate_QuantityDType(units);
}

static void
QuantityDType_dealloc(QuantityDTypeObject *self)
{
    Py_CLEAR(self->units);
    PyArrayDescr_Type.tp_dealloc((PyObject *)self);
}

static PyObject *
QuantityDType_repr(QuantityDTypeObject *self)
{
    return PyUnicode_FromFormat("QuantityDType(%R)", self->units);
}

PyArray_DTypeMeta QuantityDType = {
    {{
        PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "QuantityDType",
        .tp_basicsize = sizeof(QuantityDTypeObject),
        .tp_new = QuantityDType_new,
        .tp_dealloc = (destructor)QuantityDType_dealloc,
        .tp_repr = (reprfunc)QuantityDType_repr,
        .tp_str = (reprfunc)QuantityDType_repr,
    }},
};

int init_QuantityDType(void)
{
    PyArrayMethod_Spec *casts[] = {&CastSpec, NULL};
    PyArrayDTypeMeta_Spec QuantityDType_Spec = {
        .flags = NPY_DT_PARAMETRIC,
        .casts = casts,
        .typeobj = Quantity_Type,
        .slots = QuantityDType_Slots,
    };
    ((PyObject *)&QuantityDType)->ob_type = &PyArrayDTypeMeta_Type;
    ((PyTypeObject *)&QuantityDType)->tp_base = &PyArrayDescr_Type;
    if (PyType_Ready((PyTypeObject *)&QuantityDType) < 0)
        return -1;
    if (PyArrayInitDTypeMeta_FromSpec(&QuantityDType, &QuantityDType_Spec) < 0)
        return -1;
    QuantityDType.singleton = PyArray_GetDefaultDescr(&QuantityDType);
    return 0;
}
