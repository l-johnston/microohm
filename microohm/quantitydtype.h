#pragma once
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"

typedef struct
{
    PyArray_Descr base;
    PyObject *units;
} QuantityDTypeObject;

extern PyArray_DTypeMeta QuantityDType;
extern int init_QuantityDType(void);
extern PyObject *Quantity_Type;
QuantityDTypeObject *instantiate_QuantityDType(PyObject *units);
