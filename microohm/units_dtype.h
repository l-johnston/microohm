#pragma once
#include <Python.h>
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"

typedef struct
{
    PyArray_Descr base;
    PyObject *units;
} Units_dtypeObject;

extern PyArray_DTypeMeta Units_dtype;
Units_dtypeObject *instantiate_Units(PyObject *units);
int initialize_Units(void);
