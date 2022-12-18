#pragma once
#include <Python.h>
#include "numpy/arrayobject.h"

typedef struct
{
    PyArray_Descr base;
    PyObject *units;
} UnitsObject;

UnitsObject *new_units(PyObject *units);
