#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL QuantityDType_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/experimental_dtype_api.h"
#include "quantitydtype.h"
#include "umath.h"

static int
dd_maximum_strided_loop(
    PyArrayMethod_Context *context,
    char *const data[],
    npy_intp const dimensions[],
    npy_intp const strides[],
    NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];
    while (N--)
    {
        *(double *)out = *(double *)in1 > *(double *)in2 ? *(double *)in1 : *(double *)in2;
        in1 += in1_stride;
        in2 += in2_stride;
        out += out_stride;
    }
    return 0;
}

static NPY_CASTING
qq_maximum_resolve_descriptors(
    PyObject *self,
    PyArray_DTypeMeta *dtypes[],
    PyArray_Descr *given_descrs[],
    PyArray_Descr *loop_descrs[],
    npy_intp *unused)
{
    PyObject *units1 = ((QuantityDTypeObject *)given_descrs[0])->units;
    PyObject *units2 = ((QuantityDTypeObject *)given_descrs[1])->units;
    if (PyUnicode_Compare(units1, units2) != 0)
    {
        PyErr_SetString(PyExc_ValueError, "can only compare units of the same dimension");
        return -1;
    }
    loop_descrs[2] = instantiate_QuantityDType(units1);
    if (loop_descrs[2] == NULL)
        return -1;
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];
    return NPY_NO_CASTING;
}

int init_maximum_ufunc(void)
{
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (numpy == NULL)
        return -1;
    PyObject *maximum = PyObject_GetAttrString(numpy, "maximum");
    Py_DECREF(numpy);
    if (maximum == NULL)
        return -1;
    static PyArray_DTypeMeta *dtypes[3] = {&QuantityDType, &QuantityDType, &QuantityDType};
    static PyType_Slot slots[] = {
        {NPY_METH_resolve_descriptors, &qq_maximum_resolve_descriptors},
        {NPY_METH_strided_loop, &dd_maximum_strided_loop},
        {0, NULL}};
    PyArrayMethod_Spec qq_MaximumSpec = {
        .name = "qq_maximum",
        .nin = 2,
        .nout = 1,
        .dtypes = dtypes,
        .slots = slots,
        .flags = 0,
        .casting = NPY_NO_CASTING,
    };
    if (PyUFunc_AddLoopFromSpec(maximum, &qq_MaximumSpec) < 0)
    {
        Py_DECREF(maximum);
        return -1;
    }
}
